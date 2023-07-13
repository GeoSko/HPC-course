def scaler_resuls(scaler):
    print("="*40)
    print("==> StandardScaler <==")
    for attr in vars(scaler):
        print("scaler.{} = {}".format(attr, getattr(scaler, attr)))