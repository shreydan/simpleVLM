import torch

def generate(
    model,
    processor,
    inputs,
    max_length=50,
    deterministic=False,
    temperature=1.0,
    enable_kv_cache=True,
):
    """
    Simplified generate function for a causal VLM.

    Args:
        model: The causal VLM model
        inputs: Dict containing 'input_ids', 'pixel_values', 'attention_mask'
        max_length: Maximum length of generated sequence
        temperature: Temperature for sampling (higher = more random)
        enable_kv_cache: Whether to use KV cache for faster generation

    Returns:
        generated_sequences: Tensor of shape [batch_size, sequence_length]
    """
    # Extract inputs
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    pixel_values = inputs["pixel_values"].cuda() if inputs['pixel_values'] is not None else None


    # Initialize generated sequences with input_ids
    generated_sequences = input_ids.clone()
    current_length = generated_sequences.shape[1]

    # Setup KV cache
    if enable_kv_cache:
        model.text_model.enable_kv_cache()
        model.text_model.reset_kv_cache()
    else:
        model.text_model.disable_kv_cache()

    # First forward pass always includes all inputs
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )

    # Generate tokens one by one
    while current_length < max_length:
        # Get the next token logits and apply temperature
        next_token_logits = outputs[:, -1, :]
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Sample from the distribution
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        if deterministic:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        # Add the sampled token to the output
        generated_sequences = torch.cat([generated_sequences, next_token], dim=1)
        current_length += 1

        # Create attention mask for the new token
        token_attention_mask = torch.ones_like(next_token).bool()

        # For subsequent forward passes
        with torch.no_grad():
            if enable_kv_cache:
                # With KV cache, only process the new token
                outputs = model(
                    input_ids=next_token,
                    attention_mask=token_attention_mask
                )
            else:
                # Without KV cache, process the full sequence
                attention_mask = torch.cat([attention_mask, token_attention_mask], dim=1)
                outputs = model(
                    input_ids=generated_sequences,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask
                )

        if next_token.flatten().item() == processor.tokenizer.eos_token_id:
            break

    # Clean up KV cache if needed
    if enable_kv_cache:
        model.text_model.disable_kv_cache()

    return processor.tokenizer.decode(generated_sequences.flatten().cpu().numpy())