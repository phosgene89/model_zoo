import torch

from AdvancedConvRNN.encoder_decoder import *
from AdvancedConvRNN.net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
    
def generate_convlstm(convlstm=True, convgru=False):
    
    if convlstm:
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params
    if convgru:
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params
    else:
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params
    
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    net = EncoderDecoder(encoder, decoder)
    
    return net