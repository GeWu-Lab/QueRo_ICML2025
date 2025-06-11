import torch
import torch.nn as nn


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output



from .mmml import *
class MMML_head(nn.Module):
    def __init__(self, edim, n_classes, num_hidden_layers, dropout=0.0):
        super(MMML_head, self).__init__()
        self.name = 'MMML'
        
        feature_dim = token_dim = edim

        self.V_output_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_classes)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_classes)
          )
        
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=token_dim)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=token_dim)

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=num_hidden_layers, hidden_size=token_dim, num_attention_heads=8)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )

        self.fused_output_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, n_classes)
        )
        
    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        # elif layer_name == 'text_mixed':
        #     embedding_layer = self.text_mixed_cls_emb
        # elif layer_name == 'audio_mixed':
        #     embedding_layer = self.audio_mixed_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    # for us, resnet -> [B, 512] -> [B, seq=16, e_dim=32]
    # FOR TRANSFORMER BASED, input, v_seq, a_seq (include CLS)
    def forward(self, a, v):
        B = v.shape[0]
        V_features = v[:, 0, :]
        A_features = a[:, 0, :]

        V_token = v[:, 1:, :]
        A_token = a[:, 1:, :]

        # text output layer
        V_output = self.V_output_layers(V_features)                    # Shape is [batch_size, 2]
        
        # audio output layer
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 2]

        # TOKENIZE
        # V_features = V_features.reshape((B, 16, 32))
        V_MASK = torch.ones((B, V_token.shape[1])).to(V_features.device)
        # A_features = A_features.reshape((B, 16, 32))
        A_MASK = torch.ones((B, A_token.shape[1])).to(A_features.device)
    
        # CME layers
        ## prepend cls tokens
        text_inputs, text_attn_mask = self.prepend_cls(V_token, V_MASK, 'text') # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_token, A_MASK, 'audio') # add cls token

        # pass through CME layers
        # index = 0
        for layer_module in self.CME_layers:
            # index += 1
            # print('iter ', index)
            text_inputs, audio_inputs = layer_module(text_inputs, text_attn_mask,
                                                audio_inputs, audio_attn_mask)

        # different fusion methods
        fused_hidden_states = torch.cat((text_inputs[:,0,:], audio_inputs[:,0,:]), dim=1) # Shape is [batch_size, 768*2]
        # print(fused_hidden_states.shape)
        # last linear output layer
        fused_output = self.fused_output_layers(fused_hidden_states) # Shape is [batch_size, 2]
        
        return a, v, {
                'out_v': V_output, 
                'out_a': A_output, 
                'out': fused_output
        }


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        x = self.clf(x)
        return x

class MMdynamic(nn.Module):
    def __init__(self, edim, num_class):
        super().__init__()
        a_dim = v_dim = edim
        self.name = 'MMDynamic_RE'
        self.classes = num_class
        num_modality = 2
        self.num_modality = num_modality
        dropout = 0.1
        self.dropout = dropout

        hidden_dim = [min(a_dim, v_dim) // 4]

        self.FeatureInforEncoder_a = LinearLayer(a_dim, a_dim)
        self.TCPConfidenceLayer_a = LinearLayer(hidden_dim[0], 1) 
        self.TCPClassifierLayer_a = LinearLayer(hidden_dim[0], num_class)
        self.FeatureEncoder_a = LinearLayer(a_dim, hidden_dim[0])
                                 
        self.FeatureInforEncoder_v = LinearLayer(v_dim, v_dim)
        self.TCPConfidenceLayer_v = LinearLayer(hidden_dim[0], 1) 
        self.TCPClassifierLayer_v = LinearLayer(hidden_dim[0], num_class)
        self.FeatureEncoder_v = LinearLayer(v_dim, hidden_dim[0])

        self.MMClasifier = []
        self.MMClasifier.append(LinearLayer(self.num_modality * hidden_dim[0], hidden_dim[0]))
        self.MMClasifier.append(nn.ReLU())
        self.MMClasifier.append(nn.Dropout(p=dropout))

        for layer in range(1, len(hidden_dim)):
            self.MMClasifier.append(LinearLayer(hidden_dim[layer - 1], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        # print(self.MMClasifier)

        self.a_head = nn.Linear(edim, num_class)
        self.v_head = nn.Linear(edim, num_class)

    # input tokens!
    def forward(self, a, v):
        a = a[:, 0, :]
        v = v[:, 0, :]

        # probe
        out_a = self.a_head(a.detach().clone())
        out_v = self.v_head(v.detach().clone())

        gate_a = torch.sigmoid(self.FeatureInforEncoder_a(a))
        gate_v = torch.sigmoid(self.FeatureInforEncoder_v(v))

        a = gate_a * a
        v = gate_v * v
        a = self.FeatureEncoder_a(a)
        v = self.FeatureEncoder_v(v)

        a = F.relu(a)
        v = F.relu(v)

        a = F.dropout(a, self.dropout, training = self.training)
        v = F.dropout(a, self.dropout, training = self.training)

        TCPLogit_a = self.TCPClassifierLayer_a(a)
        TCPLogit_v = self.TCPClassifierLayer_v(v)
        # print(TCPLogit_a.shape)

        TCPConfidence_a = self.TCPConfidenceLayer_a(a)
        TCPConfidence_v = self.TCPConfidenceLayer_v(v)

        a = a * TCPConfidence_a
        v = v * TCPConfidence_v

        feature = torch.cat([v, a], dim=1)
        MM_feature = self.MMClasifier(feature)

        # USE MM_feature to cal acc
        return a, v, {
            'out': MM_feature,
            'out_a': out_a,
            'out_v': out_v,
            'gate_a': gate_a,
            'gate_v': gate_v,
            'a_conf': TCPConfidence_a,
            'v_conf': TCPConfidence_v,
            'a_log': TCPLogit_a,
            'v_log': TCPLogit_v,
        }



class QMF(nn.Module):
    def __init__(self, edim, n_classes):
        super(QMF, self).__init__()
        self.latent_dim = edim
        self.visual_clf = nn.Linear(self.latent_dim, n_classes)
        self.audio_clf = nn.Linear(self.latent_dim, n_classes)
        self.fusion_module = ConcatFusion(2 * edim, output_dim=n_classes)

    # input tokens
    def forward(self, a, v):
        a_n = a[:, 0, :]
        v_n = v[:, 0, :]

        v_out = self.visual_clf(v_n)
        a_out = self.audio_clf(a_n)

        v_energy = -torch.logsumexp(v_out, dim=1)
        a_energy = -torch.logsumexp(a_out, dim=1)

        v_conf = -0.1*torch.reshape(v_energy, (-1,1))
        a_conf = -0.1*torch.reshape(a_energy, (-1,1))

        ### DYNAMIC LATE FUSION
        out = (v_out*v_conf + a_out*a_conf)

        outputs = {
            "out": out, 
            "out_a": a_out, 
            "out_v": v_out,
            "v_conf": v_conf, 
            "a_conf": a_conf
        }
        return a, v, outputs


class SequentialEncoder(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = SequentialEncoder(
            *[TransformerEncoderLayer(input_dim, num_heads, hidden_dim)
              for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x


class JMT(nn.Module):
    def __init__(self, edim, n_classes, num_heads=12, num_layers=1):
        super(JMT, self).__init__()

        visual_dim = audio_dim = edim
        hidden_dim = edim // 16

        # Encoder blocks
        self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads,
                                                      hidden_dim, num_layers)

        self.audio_encoder = TransformerEncoderBlock(audio_dim,
                                                             num_heads,
                                                             hidden_dim,
                                                             num_layers)
        
        self.joint_representation_encoder = TransformerEncoderBlock(edim,
                                                                    num_heads,
                                                                    hidden_dim,
                                                                    num_layers)

        self.final_encoder = TransformerEncoderBlock(edim, num_heads,
                                                     hidden_dim, num_layers)

        # Cross attention
        self.cross_attention_v = nn.MultiheadAttention(visual_dim, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(audio_dim, num_heads)
        self.cross_attention_pv = nn.MultiheadAttention(edim, num_heads)

        # Fully connected layer for joint representation
        self.out_layer_pv = nn.Linear(edim * 2, edim)

        # Fully connected layer for the final output
        self.out_layer1 = nn.Linear(edim * 6, edim)

        self.a_head = nn.Linear(edim, n_classes)
        self.v_head = nn.Linear(edim, n_classes)
        self.FC = nn.Linear(edim, n_classes)
    

    def forward(self, a, v):
        out_a = self.a_head(a[:, 0, :].detach().clone())
        out_v = self.v_head(v[:, 0, :].detach().clone())

        seq_len_a = a.shape[1]
        seq_len_v = v.shape[1]

        if seq_len_a > seq_len_v:
            v = torch.cat((v, a[:, seq_len_v:seq_len_a, :]), dim=1)
        elif seq_len_a < seq_len_v:
            a = torch.cat((a, v[:, seq_len_a:seq_len_v, :]), dim=1)

        # Concatenate the visual and physiological features
        joint_representation = torch.cat((a, v), dim=2)

        # Decrease the dimensionality of the joint representation
        joint_representation = self.out_layer_pv(joint_representation)

        # Permute dimension from (batch, seq, feature) to (seq, batch, feature)
        v = v.permute(1, 0, 2)
        a = a.permute(1, 0, 2)
        joint_representation = joint_representation.permute(1, 0, 2)

        # Pass the visual, physiological and joint representation features through their respective encoders
        visual_encoded = self.visual_encoder(v)
        audio_encoded = self.audio_encoder(a)
        joint_representation_encoded = self.joint_representation_encoder(
            joint_representation)

        # Do all the cross-attention between the visual encoded and physio encoded features
        cross_attention_output_v_p, _ = self.cross_attention_v(visual_encoded, audio_encoded, audio_encoded)

        # Do all the cross-attention between the physio encoded and visio encoded features
        cross_attention_output_p_v, _ = self.cross_attention_p(audio_encoded, visual_encoded, visual_encoded)

        # Do all the cross-attention between the joint representation encoded and visio encoded features
        cross_attention_output_pv_v, _ = self.cross_attention_pv(joint_representation_encoded, visual_encoded, visual_encoded)

        # Do all the cross-attention between the visio encoded and joint representation encoded features
        cross_attention_output_v_pv, _ = self.cross_attention_v(visual_encoded, joint_representation_encoded, joint_representation_encoded)

        # Do all the cross-attention between the joint representation encoded and physio encoded features
        cross_attention_output_pv_p, _ = self.cross_attention_pv(
            joint_representation_encoded, audio_encoded,
            audio_encoded)

        # Do all the cross-attention between the physio encoded and joint representation encoded features
        cross_attention_output_p_pv, _ = self.cross_attention_p(
            audio_encoded, joint_representation_encoded,
            joint_representation_encoded)

        # Concatenate Cross-attention outputs
        concat_attention = torch.cat((cross_attention_output_v_p,
                                        cross_attention_output_p_v,
                                        cross_attention_output_pv_v,
                                        cross_attention_output_v_pv,
                                        cross_attention_output_pv_p,
                                        cross_attention_output_p_pv), dim=2)
        out = self.out_layer1(concat_attention)  # bsz, seq, edim
        '''
            REALLY THIS SHAPE???
            No, manually transpose
        '''
        out = out.transpose(0, 1)
        # print(out.shape)
        out = torch.mean(out, dim=1)
        # print(out.shape)
        out = self.FC(out)
        # print(out.shape)

        outputs = {
            "out": out, 
            "out_a": out_a, 
            "out_v": out_v,
        }
        return a, v, outputs
