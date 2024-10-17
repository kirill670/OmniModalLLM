def main():
    """
    Main function to train the model.
    """
    # ==========================================
    # Data Loading and Preparation
    # ==========================================
    
    # Loading Flickr30k dataset
    print("Loading Flickr30k dataset...")
    flickr_dataset = load_dataset("flickr30k", split='train')
    # Filter out samples without captions or images
    flickr_dataset = flickr_dataset.filter(lambda x: len(x['caption']) > 0 and x['image'] is not None)
    
    # Loading DailyDialog dataset
    print("Loading DailyDialog dataset...")
    chat_dataset = load_dataset("daily_dialog", split='train')
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Create custom Dataset instances
    flickr_custom_dataset = FlickrDataset(flickr_dataset, tokenizer.text_tokenizer, transform)
    chat_custom_dataset = ChatDataset(chat_dataset, tokenizer.text_tokenizer, max_length=512)
    
    # Determine batch size based on device
    if device.type == 'cuda':
        batch_size = 4  # Adjust as per GPU memory
    elif device.type == 'xla':
        batch_size = 2  # Adjust for TPU
    else:
        batch_size = 2  # Adjust for CPU
    
    # Create DataLoaders for both datasets
    flickr_dataloader = DataLoader(flickr_custom_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    chat_dataloader = DataLoader(chat_custom_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # ==========================================
    # Optimizer, Loss, Scheduler
    # ==========================================
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # ==========================================
    # Training
    # ==========================================
    
    print("Starting training...")
    train_model(
        model=model,
        flickr_dataloader=flickr_dataloader,
        chat_dataloader=chat_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        num_epochs=5,
        save_path='checkpoint.pth.tar',
        patience=3
    )
    print("Training completed.")
