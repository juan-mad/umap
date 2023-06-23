Definir conjunto de datasets *benchmark* para estudiar cómo varía UMAP en función de sus HPs.
 - Toy datasets (e.g. anillo y alguna otra 2-variedad), MNIST, FMNIST, ...
 - ¿Datasets de mundo real que muestren clusters o una variación "continua"?

Datasets sintéticos, que sirvan como un ground truth certero

### Opciones:

Comprobar con otros datasets los valores de la función de pérdida de UMAP vs la esperada (true loss)
- ¿Repetir experimentos de Damrich et al.? Solo lo hacen con el anillo en 2D, pero podría probarse con **otros datasets**.

~~Implementar variación de la función de pérdida propuesta por Damrich et al.
- ~~No está programada por ellos pero sí formulada. Verificar su desarrollo matemático y estudiar numéricamente si respeta las similaridades en alta dimension~~

Implementar distancia bien definida en alguna variedad (e.g. esfera, toro)
**Revisar cómo implementar distancias nuevas**
- Esfera: distancia es arcocoseno: d(x,y) = arccos(x · y)
- Toro: al ser espacio producto, la distancia es la suma por cada una de las componentes:
	$$\sqrt{\sum d_i(x_i,y_i)^2}$$ donde $d_i=d$ es la distancia en la esfera (arcocoseno)


Dataset 