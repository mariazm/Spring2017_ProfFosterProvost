import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pylab as plt

def Decision_Surface(data, target, model, surface=True, probabilities=False, cell_size=.01):
    '''
    This function creates the surface of a decision tree using the data created with this script. 
    You can change this function tu plot any column of any dataframe. 
    
    INPUT: data (created with data_tools.X() ),
            target (Y value creted with data_tools.create_data() ),
            model (Model already fitted with X and Y , i.e. DecisionTreeClassifier or logistic regression )
            surface (True if we want to display the tree surface),
            probabilities (False by default, if True we can see the color-scale based on the likelihood of being closer to the separator),
           cell_size (value for the step of the numpy arange that creates the mesh)

    RETURNS: Scatterplot with/without the surface
    '''
    # Get bounds, we only have 2 columns in the dataframe: column 0 and column 1 
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
        # Predict on the mesh with labels or probability
        if probabilities:
            Z = model.predict_proba(meshed_data)[:, 1].reshape(xx.shape)
        else:
            Z = model.predict(meshed_data).reshape(xx.shape)
    
    # Plot mesh and data
    if data.shape[1] > 2:
        # Higher orders
        plt.title("humor^(" + str(range(1,data.shape[1])) + ") and number_pets")
    else:
        plt.title("humor and number_pets")
    plt.xlabel("humor")
    plt.ylabel("number_pets")
    if surface and model != None:
        if probabilities:
            # Color-scale on the contour (surface = separator)
            cs = plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)
        else:
            # Only a curve/line on the contour (surface = separator)
            cs = plt.contourf(xx, yy, Z, levels=[-1,0,1],cmap=plt.cm.coolwarm, alpha=0.4)
    color = ["blue" if t == 0 else "red" for t in target]
    plt.scatter(data[data.columns[0]], data[data.columns[1]], color=color)

def create_data():
    '''
    This function creates a data set with 3 random normal distributions scaled. 
    There are two main variables in this dataset: humor and number_pets.
    It also computes higher orders for the 'humor' variable (^2, ^3 and ^4).
    You can change this function with new orders or column names.
   

    RETURNS: target_name (always "success"), 
             variable_names (always ["humor", "number_pets"] ), 
             data  (dataframe with the data WITHOUT higher orders), 
             Y (target variable with values 0 or 1)
    '''
    # Set the randomness
    np.random.seed(36)

    # Number of users
    n_users = 200

    # Relationships
    variable_names = ["humor", "number_pets"]
    target_name = "success"

    # Generate data (3 random normal distributions!!!!)
    a = np.random.normal(5, 5, n_users )
    b = np.random.normal(10, 5, n_users )
    c = np.random.normal(20, 5, n_users )

    # Change scales
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
    '''
    This function return the X-data from the 'create_data' function of this script.
    You can change the complexity to receive the main 2 columns + complex orders.
    
    INPUT: complexity (higher complexity (1 to 4) for the 'humor' variable)

    RETURNS: data  (dataframe with the data WITH higher orders IF required)
    '''
    # remove the target variable
    drops = ["success"]
    
    # if complexity = 1 then we just need to drop all the higher order from the dataframe
    for i in [2, 3, 4]:
        # based on the number of complexity required, we drop the rest of the higher orders
        if i > complexity:
            drops.append("humor^" + str(i))
    
    return data.drop(drops, 1)

target_name, variable_names, data, Y = create_data()