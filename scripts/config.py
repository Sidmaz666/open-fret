from transformers import T5Config

def get_tiny_tab_config(vocab_size=4000):
    """
    Returns a T5 configuration optimized for MIDI-to-Tab.
    Based on GOAL.md: d_model=128, d_ff=1024, 3 layers, 4 heads.
    """
    config = T5Config(
        vocab_size=vocab_size,
        d_model=256,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=3,
        bos_token_id=2,
        unk_token_id=1,
        decoder_start_token_id=0,
    )
    return config

if __name__ == "__main__":
    config = get_tiny_tab_config()
    print(config)
