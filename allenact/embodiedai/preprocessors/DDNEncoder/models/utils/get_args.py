vit_arc = "vit_b"

class get_brain_args():
    def __init__(self):
        # position encoding
        if(vit_arc=="vit_b"):
            self.hidden_dim = int(768)
        else:
            self.hidden_dim = int(384)
        self.position_embedding = 'sine'
        
        # dino backbone
        self.enc_output_layer = int(1) # for stream, is 1; for visual, is 8
        self.vit_arc = 'vit_b'

        # Transformer decoder
        self.dropout = float(0.1)
        self.nheads = int(16)
        self.dim_feedforward = int(2048)
        self.enc_layers = int(0)
        self.dec_layers = int(1)
        self.pre_norm = False

        # DETR
        self.readout_res = 'streams_inc'
        self.decoder_arch = 'transformer'
        self.output_layer = 'backbone'
        self.num_queries = int(16)
        self.lh_vs = int(19004)
        self.rh_vs = int(20544)