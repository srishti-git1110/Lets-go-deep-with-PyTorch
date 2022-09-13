
def train(n_epochs, cnn, train_loader):
    n_batches = len(train_loader)
    for epoch in range(n_epochs):
        for batch, (images, labels) in enumerate(train_loader):
            output = cnn(images)[0]               
            loss = loss_func(output, labels)   
            optimizer.zero_grad()
            loss.backward()           
            optimizer.step()                
            
            if (batch+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, n_epochs, batch + 1, n_batches, loss.item()))
