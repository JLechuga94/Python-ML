from IPython.display import display, Math, Latex

# What is the probability that a cliente buys a product knowing that it is a man
# What is the probability that a client is women knowing that they bought a product
display(Math(r'P(Purchase|Male) = \frac{Número total de compras hechas por hombre}{Numero total de hombres}'))
display(Math(r'P(Female|Purchase) = \frac{Número total de compras hechas por mujeres}{Numero total de compradores} = \frac{Female \cap Purchase}{Purchase}'))
