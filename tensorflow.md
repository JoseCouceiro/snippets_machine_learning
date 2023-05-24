Tensorflow

Choosing and optimizer and loss

# For a multi_class classification problem
model.compile(opimizer= 'rmsprop',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])

# For a binary classification problem
model.compile(optimizer= 'rmsprop',
              loss= 'binary_crossentropy',
              metrics= ['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer= 'rmsprop',
              loss= 'mse')