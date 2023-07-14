def scaler_resuls(scaler,type):
    print("="*40)
    if(type == "std"):
        print("==> StandardScaler <==")
    if(type == "min_max"):
        print("==> MinMaxScaler <==")
    for attr in vars(scaler):
        print("scaler.{} = {}".format(attr, getattr(scaler, attr)))