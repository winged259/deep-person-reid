import torchreid

def main():
    datamanager = torchreid.data.ImageDataManager(
        root="/Users/duynguyen/Documents/IVSR/Re-ID/dataset",
        sources="uavhuman",
        targets="uavhuman",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"])

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True)

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003)

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20)
    engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True)

    engine.run(
    save_dir="log/resnet50",
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False)

if __name__ == "__main__":
    main()