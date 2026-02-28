# NotacionBigO
Búsqueda eficiente sobre grandes volúmenes de datos
## Análisis de complejidad teórica
<img width="700" height="531" alt="image" src="https://github.com/user-attachments/assets/66fc02f2-3a26-4c6c-b1f3-d609bd0303ef" />


El **arreglo ordenado** utiliza búsqueda binaria, lo que garantiza una complejidad de **O(log n)** en tiempo de búsqueda, con bajo consumo de memoria relativa.

La **tabla hash (set)** presenta una complejidad de **O(1) en promedio**, ya que el acceso depende de una función hash. Este rendimiento se logra a costa de un mayor consumo de memoria debido a su estructura interna.

El **árbol binario de búsqueda (BST)** tiene una complejidad de **O(h)**, donde `h` es la altura del árbol. En condiciones favorables puede aproximarse a **O(log n)**, pero en el peor caso puede degradarse a **O(n)**.

El **árbol AVL** mantiene el balance mediante rotaciones, garantizando una complejidad de **O(log n)**. Sin embargo, este balance introduce un mayor costo constante en tiempo y memoria.


# Conclusion
La **tabla hash (set)** fue la estructura más eficiente en términos de **tiempo de búsqueda**, validando su complejidad teórica **O(1) promedio**.

No obstante, si se prioriza el **consumo de memoria**, el **arreglo ordenado** representa la mejor alternativa, ya que ofrece búsquedas en **O(log n)** con un uso significativamente menor de memoria.
