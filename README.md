## Repositorio Taller 2

**Facultad de Economía**

**Universidad de los Andes**

Integrantes: [David Santiago Caraballo Candela](https://github.com/scaraballoc), [Sergio David Pinilla Padilla](https://github.com/sdpinilla18) y [Juan Diego Valencia Romero](https://github.com/judval).

En el presente repositorio se encuentran todos los documentos, bases de datos y códigos utilizados durante el desarrollo del Taller 2 de la clase *Big Data & Machine Learning for Applied Economics*, del profesor [Ignacio Sarmiento-Barbieri](https://ignaciomsarmiento.github.io/igaciomsarmiento) durante el segundo semestre del año 2022.

Este trabajo tenía como objetivo el desarrollo de dos (2) modelos; i) clasificación y ii) de predicción de ingreso para determinar el estatus de pobreza de los hogares de distintas ciudades en Colombia, a partir del uso de una base de datos del 2018 de la Gran Encuesta Integrada de Hogares (GEIH) recolectada por el Departamento Administrativo Nacional de Estadistica (DANE). Tal insumo, con la intención de mejorar el proceso de identificación de hogares por debajo del umbral de pobreza. Se importa toda la *raw-database* de la GEIH 2018 que contiene un total de 542.941 observaciones y 157 variables que resulta de unificar las bases de los hogares y personas.

**1. *Data cleaning & Modelling***

La estrategia empírica del trabajo sigue el orden del objetivo. Se inició estimando nueve (9) modelos en el rubro de regresión que se listan como sigue: i) Ridge *Cross-Validation* con *default scoring*; ii) Ridge *Cross-Validation* con *Pinball scoring*; iii) Lasso; iv) *Elastic Net* con *Cross-Validation*; v) *Elastic Net* con *Pinball scoring*; vi) Cuantílica P25; vii) Cuantílica no regularizada; viii) Cuantílica *Cross-Validation*, y; ix) Mínimos Cuadrados Ordinarios. Por su parte, en el rubro de clasificación se especificaron seis (6) técnicas: i) *Quadratic Discriminant Analysis*; ii) *Linear Discriminant Analysis*; iii) Logit; iv) *K-Nearest-Neighbors*; v) *Naive Bayes*, y; vi) Random Forest.

Para poder utilizar nuestro código de **Python**, es necesario tener instalados los paquetes de `numpy`, `pyread`, `sklearn`, `pandas`, `scipy` y `matplotlib`; de los cuales se importan diversas librerías. El código completo, que incluye todo el proceso de limpieza de datos, extracción de estadísticas descriptivas y el análisis empírico para responder a las preguntas del *problem set* se encuentran en orden dentro del notebook de Jupyter titulado "PS2_BD.ipynb". El *Python script* asociado al notebook esta titulado como "T2Script.py".

***Nota:*** *Este archivo debería correr a la perfección siempre y cuando se sigan las instrucciones y comentarios del código (en orden y forma). Es altamente recomendable que antes de verificar la replicabilidad del código, se asegure de tener **todos** los requerimientos informáticos previamente mencionados (i.e. se prefieren versiones de **Python** menores a la 3.10.9 para evitar que paquetes, funciones y métodos que han sido actualizados no funcionen). Además, la velocidad de ejecución dependerá de las características propias de su máquina, por lo que deberá (o no) tener paciencia mientras se procesa.*
