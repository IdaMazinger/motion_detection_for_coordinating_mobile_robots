from ultralytics import YOLO

RESUME = False


def resume_training() -> None:
    """Resume existing algorithm training"""
    model = YOLO('<path_to_runs_detect_train_weights>/last.pt')
    results = model.train(resume=True)
    print(results)


def train_new_model() -> None:
    """Train existing or new model"""
    model = YOLO('<path_to_model>/yolo_model.pt')
    results = model.train(
        data='<path_to_dataset>/data.yaml',
        epochs=10
    )
    print(results)


if __name__ == '__main__':
    if RESUME:
        resume_training()
    else:
        train_new_model()
