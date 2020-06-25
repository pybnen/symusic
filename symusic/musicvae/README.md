# MusicVAE

Run the following command to train a model.

```bash
python -m symusic.musicvae.train with /path/to/config -F /path/to/log/runs
```

## Configs

Example configurations can be seen here:
- [flat config](/symusic/musicvae/config_flat_8bar.json)
- [hierarchical config](/symusic/musicvae/config_hier_8bar.json)

**Data Loader**  
`batch_size`: Size of batch returned by the data loader  
`num_workers`: Number of data loader worker processes  

**Dataset**  
`train_dir` Directory to midi files used for training  
`eval_dir`  Directory to midi files used for evaluation  
`slice_bar` Length of melody in bars  


**Model**  
`ckpt_path` Continue training from this checkpoint  

`z_size`  Size of the z space  

**Encoder**  
`encoder_params`             Parameter of encoder  
`encoder_params.embed_size`  Output size of embedding  
`encoder_params.hidden_size` Hidden size  
`encoder_params.n_layers`    Number of LSTM layers  

**Sequential decoder**  
`seq_decoder_args`           Arguments for the seq decoder factory  
`seq_decoder_args.key`       Type of sequential decoder, available are 'sample', 'greedy' and 'hier'  
`seq_decoder_args.params`    Parameter for the sequential decoder, depends on the type  

*Params for greedy:*  
The greedy decoder uses the most likely output token as next input for the decoder.  

`z_size`       The context size of the encoder (or conductor if used as bottom layer decoder in hier decoder)  
`decoder_args` Arguments for the decoder (see *decoder_args*)  

*Params for sample:*  
The sample decoder samples from the output distribution to get the next input for the decoder.  

`z_size`       The context size of the encoder (or conductor if used as bottom layer decoder in hier decoder)
`temperature`  Parameter used to flatten/steepen the output distribution (default 1.0)  
`decoder_args` Arguments for the decoder (see *decoder_args*)  

*Params for hier:*  
The hier decoder has a high-level conductor that provides a context for the bottom level decoder
which decodes subsequences of a given sequence.  

`z_size`           The context size of the encoder (or conductor if used as bottom layer decoder in hier decoder)  
`n_subsequences`   Number of subsequences  
`conductor_args`   Arguments for the conductor (see *decoder_args*)  
`seq_decoder_args` Arguments for the bottom layer seq decoder this is again a seq_decoder (see *seq_decoder_args*)

**Decoder**  
Defines the decoder to be used in a sequential decoder

`decoder_args`           Arguments for the decoder factory  
`decoder_args.key`       Type of decoder, available are 'simple', 'another' and 'conductor'  
`decoder_args.params`    Parameter for the decoder, depends on the type  


*Params for simple:*  
Simple decoder gets previous hidden state and input token and returns logits for next output.  

`embed_size`             Embedding size of input token  
`hidden_size`            Hidden size  
`n_layers`               Number of LSTM layers  

*Params for another:*  
This decoder also uses the context as input to predict the next output.

`embed_size`             Embedding size of input token  
`hidden_size`            Hidden size  
`n_layers`               Number of LSTM layers  
`z_size`                 Size of current context

*Params for conductor:*  
Only be used as conductor in hier sequential decoder  

`hidden_size`            Hidden size  
`n_layers`               Number of LSTM layers  
`c_size`                 Size of conductor output


**Training process**  
`num_steps` Number of steps to train  

`learning_rate`           Learning rate of optimizer  
`lr_scheduler_factor`     The factor of which the lr scheduler should reduce the learning rate  
`lr_scheduler_patience`   Number of evaluation steps, with no improvements, after which the lr scheduler should
reduce the learning rate  

`sampling_settings`          Settings for the input sampling rate of the decoder  
`sampling_settings.schedule` "constant" or "inverse_sigmoid"  
`sampling_settings.rate`     Meaning depends on schedule, e.g. schedule "constant" and "rate" 0.0 means teacher forcing  

`beta_settings`            Settings for the constant term of the KL divergence  
`beta_settings.start_beta` Constant term at the start  
`beta_settings.end_beta`   Constant term at the end  
`beta_settings.duration`   The duration (in steps) over which the constant term should increase      
    
`free_bits`                KL divergence must be greater than free bits to contribute to the loss  

**Logging**  
`evaluate_interval` Number of steps after which to evaluate model  
`advanced_interval` Number of steps after which (maybe) more costly logging is done  
`print_interval`    Number of steps to print current loss


