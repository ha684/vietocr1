import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
                
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src: src_len x batch_size x img_channel
        src_lengths: batch_size (lengths of each sequence in the batch)
        outputs: src_len x batch_size x hid_dim 
        hidden: batch_size x hid_dim
        """
        embedded = self.dropout(src)
        # Pack padded sequence
        outputs,hidden = self.rnn(embedded)
        # Concatenate the hidden states from both directions
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, dec_hid_dim]
        encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        attention: [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, dec_hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, encoder_outputs):
        """
        trg: [batch_size, trg_len]
        hidden: [batch_size, dec_hid_dim]
        encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        outputs: [batch_size, trg_len, output_dim]
        """
        # Embed the entire target sequence
        embedded = self.dropout(self.embedding(trg))  # [batch_size, trg_len, emb_dim]

        # Initialize tensors to hold outputs
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.output_dim).to(embedded.device)

        for t in range(trg_len):
            # Get the embedding of the current input word
            embedded_t = embedded[:, t, :].unsqueeze(1)  # [batch_size, 1, emb_dim]

            # Compute attention weights
            a = self.attention(hidden, encoder_outputs)  # [batch_size, src_len]
            a = a.unsqueeze(1)  # [batch_size, 1, src_len]

            # Compute the weighted sum of encoder outputs
            weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, enc_hid_dim * 2]

            # Concatenate embedded input word and weighted encoder outputs
            rnn_input = torch.cat((embedded_t, weighted), dim=2)  # [batch_size, 1, emb_dim + enc_hid_dim * 2]

            # Pass through RNN
            output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))  # output: [batch_size, 1, dec_hid_dim]

            # Compute predictions
            output = output.squeeze(1)  # [batch_size, dec_hid_dim]
            weighted = weighted.squeeze(1)  # [batch_size, enc_hid_dim * 2]
            embedded_t = embedded_t.squeeze(1)  # [batch_size, emb_dim]
            prediction = self.fc_out(torch.cat((output, weighted, embedded_t), dim=1))  # [batch_size, output_dim]

            # Store predictions
            outputs[:, t, :] = prediction

        return outputs, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, encoder_hidden, decoder_hidden, img_channel, decoder_embedded, dropout=0.1):
        super().__init__()
        
        attn = Attention(encoder_hidden, decoder_hidden)
        
        self.encoder = Encoder(img_channel, encoder_hidden, decoder_hidden, dropout)
        self.decoder = Decoder(vocab_size, decoder_embedded, encoder_hidden, decoder_hidden, dropout, attn)
    
    def forward_encoder(self, src):       
        """
        src: timestep x batch_size x channel
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        encoder_outputs, hidden = self.encoder(src)

        return (hidden, encoder_outputs)

    def forward_decoder(self, tgt, memory):
        """
        tgt: timestep x batch_size 
        hidden: batch_size x hid_dim
        encouder: src_len x batch_size x hid_dim
        output: batch_size x 1 x vocab_size
        """
        
        tgt = tgt[-1]
        hidden, encoder_outputs = memory
        output, hidden, _ = self.decoder(tgt, hidden, encoder_outputs)
        output = output.unsqueeze(1)
        
        return output, (hidden, encoder_outputs)

    def forward(self, src, trg):
        """
        src: src_len x batch_size x img_channel
        trg: trg_len x batch_size
        outputs: batch_size x trg_len x output_dim
        """

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)

        # Decode the target sequence
        outputs, _ = self.decoder(trg, hidden, encoder_outputs)  # outputs: [trg_len, batch_size, output_dim]

        # Transpose outputs to [batch_size, trg_len, output_dim]
        outputs = outputs.permute(1, 0, 2).contiguous()

        return outputs
    
    def decode(self, trg, hidden, encoder_outputs):
        # This method will be called by DataParallel and handles the per-GPU batch
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        device = trg.device

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

        for t in range(trg_len):
            input = trg[t]
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output

        return outputs
    
    def expand_memory(self, memory, beam_size):
        hidden, encoder_outputs = memory
        hidden = hidden.repeat(beam_size, 1)
        encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)

        return (hidden, encoder_outputs)
    
    def get_memory(self, memory, i):
        hidden, encoder_outputs = memory
        hidden = hidden[[i]]
        encoder_outputs = encoder_outputs[:, [i],:]

        return (hidden, encoder_outputs)
