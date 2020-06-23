# MusicVAE

Run the following command to train a model.

```bash
python -m symusic.musicvae.train with /path/to/config -F /path/to/log/runs
```

## Configs

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

`encoder_params`             Parameter of encoder  
`encoder_params.embed_size`  Output size of embedding  
`encoder_params.hidden_size` Hidden size  
`encoder_params.n_layers`    Number of LSTM layers  

`use_hier` Flag whether to use hierarchical or flat decoder

`decoder_params`             Parameter of (bottom) decoder  
`decoder_params.embed_size`  Output size of embedding  
`decoder_params.hidden_size` Hidden size  
`decoder_params.n_layers`    Number of LSTM layers  

*hierarchical decoder only*:  
`n_subsequences` Number of subsequences  

`c_size` Size of conductor embedding  

`conductor_params`             Parameter of conductor  
`conductor_params.hidden_size` Hidden size  
`conductor_params.n_layers`    Number of LSTM layers  

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


