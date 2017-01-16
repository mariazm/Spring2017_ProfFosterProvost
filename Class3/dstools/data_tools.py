import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pylab as plt

def Decision_Surface(data, target, model, surface=True, probabilities=False, cell_size=.01):
    # Get bounds
    x_min, x_max = data[data.columns[0]].min(), data[data.columns[0]].max()
    y_min, y_max = data[data.columns[1]].min(), data[data.columns[1]].max()
    
    # Create a mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_size), np.arange(y_min, y_max, cell_size))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    
    # Add interactions
    for i in range(data.shape[1]):
        if i <= 1:
            continue
        
        meshed_data = np.c_[meshed_data, np.power(xx.ravel(), i)]

    if model != None:
        # Predict on the mesh
        if probabilities:
            Z = model.predict_proba(meshed_data)[:, 1].reshape(xx.shape)
        else:
            Z = model.predict(meshed_data).reshape(xx.shape)
    
    # Plot mesh and data
    if data.shape[1] > 2:
        plt.title("humor^(" + str(range(1,data.shape[1])) + ") and number_pets")
    else:
        plt.title("humor and number_pets")
    plt.xlabel("humor")
    plt.ylabel("number_pets")
    if surface and model != None:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    color = ["blue" if t == 0 else "red" for t in target]
    plt.scatter(data[data.columns[0]], data[data.columns[1]], color=color)

def create_data():
    # Set the randomness
    np.random.seed(36)

    # Number of users
    n_users = 600

    # Relationships
    variable_names = ["humor", "number_pets"]
    target_name = "success"

    # Generate data (3 random normal distributions!!!!)
    a = np.random.normal(5, 5, n_users )
    b = np.random.normal(10, 5, n_users )
    c = np.random.normal(20, 5, n_users )

    x1 = list(a+10) + list(c+10) + list(b+10)
    x2 = list((b+10)/10) + list((b+10)/10) + list((c+10)/10)
    target = list(np.ones(len(b))) + list(np.ones(len(b))) + list(np.zeros(len(b)))

    data = pd.DataFrame(np.c_[x1, x2], columns=variable_names)

    # Add interactions
    data['humor^2'] = np.power(data['humor'], 2)
    data['humor^3'] = np.power(data['humor'], 3)
    data['humor^4'] = np.power(data['humor'], 4)

    data[target_name] = target

    Y = data[target_name]
    return target_name, variable_names, data, Y

def X(complexity=1):
    drops = ["success"]
    
    for i in [2, 3, 4]:
        if i > complexity:
            drops.append("humor^" + str(i))
    
    return data.drop(drops, 1)

target_name, variable_names, data, Y = create_data()