def test(cnn, test_loader):
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            test_output = cnn(images)[0]
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum()
            total += labels.size(0)
    accuracy = (float(correct)/float(total))*100     
    print('Test Accuracy of the model : %.2f' % accuracy)
