"""

GORST: "generation of rhythm-game charts via song-to-token mapping".

Transformer encoder-decoder.

input: one-bar-long mel-spectrogram, shaped as follows -> (time 6*48, freq 80)
output(target): tokenized one-bar-long chart 
loss: plain ol' cross-entropy
"""
import torch 
import torch.nn as nn



class GorstFineDiffInDecoderInput(nn.Module):
    def __init__(self, 
                 encoder_hidden_size=512,
                 decoder_hidden_size=512,
                 encoder_layers=6,
                 decoder_layers=6,
                 decoder_max_length=60,
                 condition_dim=64,
                 vocab_size=178,
                 norm_first=False,
                 bidirectional_past_context=False,
                 dropout=0.1,
                 initialize_method="uniform", initialize_std=0.1):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        assert self.encoder_hidden_size == self.decoder_hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.decoder_max_length = decoder_max_length
        self.vocab_size=vocab_size
        self.condition_dim = condition_dim
        self.bidirectional_past_context = bidirectional_past_context
        self.dropout = dropout

        self.decoder_conditions = nn.Sequential(
            nn.Linear(1,condition_dim*2),
            nn.ReLU(),
            nn.Linear(condition_dim*2,condition_dim),
            nn.ReLU()
        )

        self.encoder_embedding = nn.Conv1d(80, self.encoder_hidden_size, 1)
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.decoder_hidden_size - self.condition_dim)

        self.encoder_positional_embedding = nn.Parameter(data=torch.ones([1, 4*48, self.encoder_hidden_size], dtype=torch.float32))
        self.decoder_positional_embedding_generator = nn.Embedding(self.decoder_max_length, self.decoder_hidden_size) 
        self.decoder_positional_embedding_seeds = torch.arange(self.decoder_max_length,device='cuda:0')

        self.final = nn.Conv1d(self.decoder_hidden_size, self.vocab_size, 1)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.encoder_hidden_size, nhead=8, batch_first=True, norm_first=norm_first, dropout=dropout),
            num_layers=self.encoder_layers
        )
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=self.decoder_hidden_size, nhead=8, batch_first=True,norm_first=norm_first, dropout=dropout)
            for _ in range(self.decoder_layers)
        ])

        self.initialize(initialize_method, initialize_std)

    def initialize(self, init_method, std):
        if init_method == "uniform":
            initrange = std
            self.decoder_embedding.weight.data.uniform_(-initrange,initrange)
            self.decoder_positional_embedding_generator.weight.data.uniform_(-initrange,initrange)
            torch.nn.init.uniform_(self.encoder_positional_embedding, a=-initrange, b=initrange)
        elif init_method == "normal":
            self.decoder_embedding.weight.data.normal_(std=std)
            self.decoder_positional_embedding_generator.weight.data.normal_(std=std)
            torch.nn.init.normal_(self.encoder_positional_embedding, mean=0.0,std=std)
        elif init_method == "trunc_normal":
            torch.nn.init.trunc_normal_(self.decoder_embedding.weight, std=std,a=-2*std,b=2*std)
            torch.nn.init.trunc_normal_(self.decoder_positional_embedding_generator.weight, std=std,a=-2*std,b=2*std)
            torch.nn.init.trunc_normal_(self.encoder_positional_embedding, mean=0.0,std=std,a=-2*std,b=2*std)

    def generate_square_subsequent_mask(self, sz):
        if not self.bidirectional_past_context:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask
        else:
            mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
            mask[:-100,:-100] = True 
            #
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self,spec,tgt,cond):
        """
        params
        spec: (B, 80, 4*48)
        tgt: (B, T)
        cond: (B, 1)
        """
        encoded = self.encoder_embedding(spec).permute((0,2,1))
        encoded += self.encoder_positional_embedding

        cond = self.decoder_conditions(cond) # (B, self.condition_dim)
        cond = cond.unsqueeze(1)
        cond = cond.repeat(1,tgt.shape[1],1)

        tgt_in = torch.cat((self.decoder_embedding(tgt),cond),dim=2)
        target_length = tgt.shape[1]
        tgt_in += self.decoder_positional_embedding_generator(self.decoder_positional_embedding_seeds[:target_length]).unsqueeze(0)
        contexts = self.encoder(encoded)
        decoder_output = tgt_in
        square_mask = self.generate_square_subsequent_mask(target_length).to('cuda:0')
        for n, decoder_layer in enumerate(self.decoder):
            decoder_output = decoder_layer(tgt=decoder_output, memory=contexts, tgt_mask=square_mask)

        logits = self.final(decoder_output.permute(0,2,1)).permute((0,2,1))
        
        return logits 

if __name__=="__main__":
    net = GorstFineDiffInDecoderInput(bidirectional_past_context=True).to('cuda')
    input = torch.ones([32,80,4*48]).to('cuda')
    target = torch.ones([32,24],dtype=torch.long).to('cuda')
    condition = torch.ones([32,1]).to('cuda')
    print(net(input,target,condition).shape)

