import numpy as np
from sklearn import datasets
import pandas as pd

def Decision_Surface(data, target, model, surface=True, cell_size=.01):
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

    # Predict on the mesh
    Z = model.predict(meshed_data).reshape(xx.shape)
    
    # Plot mesh and data
    if data.shape[1] > 2:
        plt.title("humor^(" + str(range(1,complexity+1)) + ") and number_pets")
    else:
        plt.title("humor and number_pets")
    plt.xlabel("humor")
    plt.ylabel("number_pets")
    if surface:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.cool, alpha=0.3)
    color = ["blue" if t == 0 else "red" for t in target]

def create_data():
    # Set the randomness
    np.random.seed(36)

    # Number of users
    n_users = 300

    # Relationships
    variable_names = ["humor", "number_pets", "age", "favorite_number"]
    target_name = "success"

    # Generate data
    predictors, target = datasets.make_classification(n_features=4, n_redundant=0, 
                                                      n_informative=2, n_clusters_per_class=2,
                                                      n_samples=n_users)
    data = pd.DataFrame(predictors, columns=variable_names)

    # Scale
    data['humor'] = data['humor'] * 10 + 50
    data['number_pets'] = (data['number_pets'] + 6)/2

    # Add interactions
    data['humor^2'] = np.power(data['humor'], 2)
    data['humor^3'] = np.power(data['humor'], 3)

    data[target_name] = target

    Y = data[target_name]
    return target_name, variable_names, data, Y

def X(complexity=1):
    drops = ["success", "age", "favorite_number"]
    
    for i in [2, 3]:
        if i > complexity:
            drops.append("humor^" + str(i))
    
    return data.drop(drops, 1)
    
target_name, variable_names, data, Y = create_data()