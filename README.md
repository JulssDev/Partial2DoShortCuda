# Entrenamiento de Redes Neuronales en GPU
* CUDA con PyTorch en Google Colab

---

| | |
|---|---|
| **Parcial** | Segundo Corte |
| **Materia** | Programación Paralela y Computación Distribuida |
| **Profesor** | |
| **Integrantes** | |
| | |
| **Fecha** | |

---

## 0. Instrucciones Generales
* El taller se desarrolla en Google Colab usando una GPU gratuita de NVIDIA.
* Se trabaja en parejas; ambos integrantes deben entender cada parte.
* Se deben capturar pantallazos de cada salida importante indicada con [PANTALLAZO].
* Al finalizar, se descarga el notebook y se sube todo a un repositorio de GitHub.

### Preguntas
1. ¿Qué diferencia hay entre un notebook en la nube (Colab) y un entorno local como el del tutorial de instalación? ¿Cuál prefieren y por qué?
Dentro de un entorno local podemos seleccionar nuestra grafica que utilice nuestro dispositivo, mientras que en el entorno de Google Collab usa una grafica especifica distribuida por este entorno, los tiempos de respuesta varian por la velocidad del internet, mientras que localmente varia segun la velocidad de la grafica que estemos usando, preferimos la version local ya que tenemos una grafica relativamente rapida, pero Google collab es una buena opcion para las personas que no dispongan de una tarjeta grafica.
2. Antes de comenzar, hagan una predicción: ¿cuántas veces más rápida creen que será la GPU comparada con la CPU en el entrenamiento? Anoten su predicción aquí y compárenla al final con el resultado real.
Almenos deberia ser unas 2 veces mas rapida.
---

## 1. Configurar el Entorno en Google Colab
* Activar la GPU desde el menú de Colab: Entorno de ejecución > Cambiar tipo de entorno de ejecución.
* Verificar que PyTorch reconoce la GPU y mostrar el nombre del dispositivo.
* Ejecutar `nvidia-smi` para ver el estado de la GPU, igual que en el tutorial de instalación.

### Preguntas
1. La salida de `nvidia-smi` muestra campos como *Driver Version*, *Memory Usage* y *GPU-Util*. ¿Qué indica cada uno?
Driver Version muestra la version de Nvidia que esta instalada en el entorno de Google Collab, Memory Usage indica la cantidad de uso de VRAM de la tarjeta grafica actualmente y cuanta VRAM hay disponible, y por ultimo GPU-Util indica el uso del procesamiento grafico de la GPU.   
2. Cuando activan el acelerador en Colab, ¿qué creen que ocurre físicamente? ¿La GPU está en su computador o en otro lugar? Propongan una analogía con algo de la vida cotidiana.
Collab solamente mostrara la opcion de usar una CPU local si esta disponible, pero directamente usa una GPU de sus propios servidores, por lo que la GPU estaria en otro lugar. Imaginemos que tenemos un carro, este queda varado en tu casa y necesitas llevarlo al taller, tu tienes otro carro, entonces puedes decidir llevarlo arrastrado con tu otro carro usando una cuerda, o llamar un servicio de gruas para que lleve tu carro.
3. `torch.cuda.is_available()` retorna `True` o `False`. ¿Qué condiciones deben cumplirse para que retorne `True`? Listen al menos tres requisitos.
Este retorna True, para que este retorne TRUE necesitamos tener una GPU NVIDIA que tenga compatibilidad con CUDA, tener instalados los drivers de CUDA y tener CUDA Toolkit compatible.
---

## 2. Conceptos: CPU vs GPU en PyTorch
* Comparar las operaciones de CUDA en C con su equivalente en PyTorch.
* Entender cómo se mueven tensores entre CPU y GPU con `.to('cuda')`.
* Definir el dispositivo al inicio del proyecto para que el código funcione con o sin GPU.

### Preguntas
1. En el tutorial anterior usaron `cudaMemcpy` para mover datos entre CPU y GPU. En PyTorch eso se hace con `.to('cuda')`. ¿Qué ventaja le ven a la forma de PyTorch? ¿Qué se pierde al abstraerlo tanto?
Pues podemos ver que al usar cudaMemcpy(destino, recepcion, tipo de dato, operacion), es una estructura mucho mas larga, en donde solo con .to('cuda') tenemos una estructura mas simplificada y facil de recordar, al igual que el manejo de memoria es mucho mas simplificado, dentro de Pytorch podemos ver que tenemos mucho menos control interno de la memopria gracias a su simplicidad de su estructura, mientras que en CUDA tenemos una estructura mucho mas especifica donde podemos encontrar opciones mas avanzadas.
2. Diagramen en Excalidraw el flujo de un tensor desde que se crea en CPU hasta que se opera en GPU y el resultado vuelve a CPU. Etiqueten cada flecha con la operación de PyTorch correspondiente.
3. ¿Por qué es una buena práctica usar la variable `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` en lugar de escribir `'cuda'` directamente en el código?

---

## 3. Preparar los Datos: Dataset MNIST
* Descargar el dataset MNIST: 60,000 imágenes de entrenamiento y 10,000 de prueba.
* Aplicar transformaciones para convertir las imágenes a tensores y normalizarlas.
* Visualizar una muestra del dataset para entender qué se va a clasificar.

### Preguntas
1. El dataset se divide en 60,000 imágenes de entrenamiento y 10,000 de prueba. ¿Por qué no se entrena con todas las 70,000? Propongan una analogía con estudiar para un examen.
2. El `DataLoader` carga los datos en lotes (*batches*) de 64 imágenes. ¿Por qué no se pasan todas las imágenes de una sola vez a la GPU? Relacionen su respuesta con el concepto de memoria que vieron en `nvidia-smi`.
3. Cada imagen tiene forma `[1, 28, 28]`. Diagramen en Excalidraw qué representa cada dimensión y cómo luce ese tensor visualmente.

---

## 4. Construir la Red Neuronal
* Definir la arquitectura: capa de entrada (784), dos capas ocultas (256 y 128), capa de salida (10 dígitos).
* Mover el modelo a la GPU con `.to(device)`.
* Contar el total de parámetros entrenables de la red.

### Preguntas
1. Diagramen en Excalidraw la arquitectura completa de la red: entrada → capa 1 → capa 2 → salida. Indiquen el número de neuronas en cada capa y qué función de activación se usa entre ellas.
2. ¿Por qué la capa de entrada tiene exactamente 784 neuronas y la de salida exactamente 10? ¿Qué pasaría si pusieran 11 neuronas en la salida?
3. Cuando hacen `modelo.to(device)`, ¿qué creen que se está transfiriendo a la GPU? ¿Es solo el código, o algo más? Propongan una analogía con el tutorial de CUDA en C.

---

## 5. Entrenar el Modelo: CPU vs GPU
* Entrenar el mismo modelo dos veces: primero en CPU, luego en GPU.
* Medir el tiempo de entrenamiento en cada dispositivo.
* Comparar los resultados y calcular cuántas veces más rápida fue la GPU.

**Código de Entrenamiento con perdida**

```python
def entrenar_con_loss(modelo, train_loader, test_loader, dispositivo, title, epocas=3):
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)

    historico_train = []
    historico_test  = []

    modelo.train()
    inicio = time.time()

    for epoca in range(epocas):
        # --- Training loss ---
        modelo.train()
        loss_train = 0
        for imagenes, etiquetas in train_loader:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            prediccion = modelo(imagenes)
            perdida = criterio(prediccion, etiquetas)

            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            loss_train += perdida.item()

        # --- Test loss ---
        modelo.eval()
        loss_test = 0
        with torch.no_grad():
            for imagenes, etiquetas in test_loader:
                imagenes = imagenes.to(dispositivo)
                etiquetas = etiquetas.to(dispositivo)
                prediccion = modelo(imagenes)
                perdida = criterio(prediccion, etiquetas)
                loss_test += perdida.item()

        avg_train = loss_train / len(train_loader)
        avg_test  = loss_test  / len(test_loader)

        historico_train.append(avg_train)
        historico_test.append(avg_test)

        print(f"Epoca {epoca+1}/{epocas} - Train loss: {avg_train:.4f} | Test loss: {avg_test:.4f}")

    tiempo = time.time() - inicio

    # --- Graficar ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epocas+1), historico_train, label='Training loss', linewidth=2)
    plt.plot(range(1, epocas+1), historico_test,  label='Test loss',     linewidth=2, linestyle='--')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.title(f'Curva de Aprendizaje {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return historico_train, historico_test, tiempo
```

### Preguntas
1. Registren aquí los tiempos obtenidos. ¿El resultado coincidió con la predicción que hicieron en la sección 0? ¿Qué los sorprendió?
2. El entrenamiento repite el ciclo: *predicción → error → ajuste de pesos*. Propongan una analogía con algo cotidiano que siga el mismo ciclo de mejora por repetición.
3. ¿Por qué creen que la GPU es más rápida en esta tarea? Relacionen su respuesta con el concepto de hilos y bloques que vieron en el tutorial de CUDA en C.

### Análisis de la Curva de Aprendizaje

Antes de responder, observen su gráfica generada y usen esta escala para interpretar el Loss:

| Loss final | Interpretación |
|---|---|
| 1.0 o más | La red no aprendió nada, está adivinando al azar |
| 0.3 - 0.5 | Aprendiendo, pero todavía comete muchos errores |
| 0.1 - 0.2 | Bien, la red entiende el problema |
| 0.07 o menos | Muy bien, la red generaliza correctamente |
| 0.01 o menos | Casi perfecto |

**Analogía:** el Training loss son los errores practicando con ejercicios del libro que ya conocen. El Test loss son los errores en el examen real, con preguntas que nunca vieron. Al inicio la red falla mucho con los ejercicios porque no sabe nada, pero como tampoco ha memorizado nada raro, falla de forma pareja en el examen. Conforme avanza, domina los ejercicios y eso se traduce en mejora en el examen real — ahí es donde las dos líneas convergen.

### Preguntas

1. Según la escala, ¿en qué rango quedó el Loss final de su modelo? ¿Lo consideran un buen resultado para 3 épocas? Justifiquen con base en la gráfica que generaron.
2. Observen en qué época convergen las dos líneas. ¿Qué creen que pasaría si entrenaran 2 épocas más — el loss seguiría bajando indefinidamente o en algún punto se detendría? ¿Qué riesgo aparece si se entrena demasiado?

---

## 6. Evaluar y Visualizar Resultados
* Calcular la precisión del modelo sobre los datos de prueba que nunca vio durante el entrenamiento.
* Visualizar predicciones reales con indicadores de acierto (verde) y error (rojo).

### Preguntas
1. ¿Por qué la precisión se mide sobre datos que el modelo nunca vio durante el entrenamiento y no sobre los mismos datos con los que aprendió?
2. Observen los dígitos que el modelo clasificó mal. ¿Tienen algo en común? ¿Por qué creen que la red se equivocó en esos casos específicos?
3. Si quisieran mejorar la precisión del modelo, ¿qué cambiarían de la arquitectura o del entrenamiento? Propongan al menos dos modificaciones y justifiquen cada una.

---

## 7. Prueba tu Propio Dígito
* Dibujar un dígito del 0 al 9 en Paint (o cualquier editor), guardarlo como imagen.
* Subir la imagen a Colab y preprocesarla para que tenga el mismo formato que MNIST: escala de grises, fondo negro, trazo blanco, tamaño 28x28.
* Pasarla al modelo entrenado y ver qué predice.
* Visualizar la imagen tal como la ve la red antes de hacer la predicción.

**codigo ejemplo**
```python
from google.colab import files
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def procesar_imagen(nombre):
    original = Image.open(nombre).convert('L')
    
    # 1. Recortar bordes (quita sombras y bordes de hoja)
    w, h = original.size
    recortada = original.crop((w*0.05, h*0.05, w*0.95, h*0.95))
    
    # 2. Aumentar contraste para separar trazo del fondo
    from PIL import ImageEnhance, ImageFilter
    contraste = ImageEnhance.Contrast(recortada).enhance(3.0)
    
    # 3. Suavizar ruido de arrugas
    suavizada = contraste.filter(ImageFilter.MedianFilter(size=3))
    
    # 4. Invertir colores (fondo negro, trazo blanco como MNIST)
    invertida = ImageOps.invert(suavizada)
    engrosada = invertida.filter(ImageFilter.MaxFilter(size=3))
    
    # 5. Escalar a 28x28
    procesada = engrosada.resize((28, 28), Image.LANCZOS)
    
    # Visualizar las etapas
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    axes[0].imshow(recortada, cmap='gray')
    axes[0].set_title('1. Recortada')
    axes[0].axis('off')
    
    axes[1].imshow(contraste, cmap='gray')
    axes[1].set_title('2. Contraste')
    axes[1].axis('off')
    
    axes[2].imshow(invertida, cmap='gray')
    axes[2].set_title('3. Invertida')
    axes[2].axis('off')
    
    axes[3].imshow(np.array(procesada), cmap='gray')
    axes[3].set_title('4. Lo que ve la red (28x28)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return procesada

subido = files.upload()
nombre = list(subido.keys())[0]

imagen = procesar_imagen(nombre)

# Pasar al modelo
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

tensor = transform(imagen).unsqueeze(0).to('cuda')

modelo_gpu.eval()
with torch.no_grad():
    salida = modelo_gpu(tensor)
    prediccion = salida.argmax(dim=1).item()

print(f'El modelo GPU predice: {prediccion}')

modelo_cpu.eval()
with torch.no_grad():
    tensor_cpu = transform(imagen).unsqueeze(0).to('cpu')
    salida = modelo_cpu(tensor_cpu)
    prediccion = salida.argmax(dim=1).item()

print(f'El modelo CPU predice: {prediccion}')
```


### Preguntas
1. ¿El modelo acertó con tu dígito dibujado a mano? Si falló, ¿por qué creen que se equivocó? Comparen su imagen con las del dataset MNIST — ¿se ven similares o muy diferentes?
2. El preprocesamiento invierte los colores de la imagen (`ImageOps.invert`). ¿Por qué es necesario hacer eso antes de pasarla al modelo? ¿Qué pasaría si no se hiciera?
3. Prueben con un dígito que crean que va a fallar — por ejemplo un 4 o un 9 escritos de forma poco convencional. ¿Falló? ¿Qué dice eso sobre las limitaciones del modelo entrenado solo con MNIST?
4. Tomar captura, de almenos una predicción que se haya hecho correctamente.


Va justo después de la celda que compara CPU vs GPU. El enunciado:

---

### Bonus: ¿Qué tan seguro está el modelo?

Hasta ahora sabemos *qué* predice el modelo, pero no *qué tan seguro* está de su respuesta. Un modelo puede predecir "7" con un 95% de confianza o con un 40% — y eso hace toda la diferencia.

Ejecuten la siguiente celda para ver la distribución de probabilidades sobre los 10 dígitos para ambos modelos. Si el modelo está seguro, un dígito tendrá un porcentaje muy alto y los demás estarán cerca de 0. Si está dudando, verán los porcentajes distribuidos entre varios dígitos.

```python
# Ver qué tan seguro está cada modelo
import torch.nn.functional as F

with torch.no_grad():
    # GPU
    tensor_gpu = transform(imagen).unsqueeze(0).to('cuda')
    salida_gpu = modelo_gpu(tensor_gpu)
    prob_gpu = F.softmax(salida_gpu, dim=1)[0]
    
    # CPU
    tensor_cpu = transform(imagen).unsqueeze(0).to('cpu')
    salida_cpu = modelo_cpu(tensor_cpu)
    prob_cpu = F.softmax(salida_cpu, dim=1)[0]

print("Probabilidades GPU:")
for i, p in enumerate(prob_gpu):
    print(f"  {i}: {p.item()*100:.1f}%")

print("\nProbabilidades CPU:")
for i, p in enumerate(prob_cpu):
    print(f"  {i}: {p.item()*100:.1f}%")
```

**Observen y respondan:**
1. ¿Cuál dígito tiene la probabilidad más alta en cada modelo? ¿Coincide con la predicción?
2. ¿El modelo está seguro o dudando? ¿Cómo lo saben mirando los porcentajes?
3. Si el porcentaje más alto es menor al 50%, ¿confiarían en esa predicción? ¿Por qué?

---

El bloque de código lo reemplazas con la función completa que ya tenemos. ¿Lo agregamos también al markdown del taller?


## 8. Preguntas de Reflexión y Entregables
* Responder 4 preguntas que conectan lo aprendido en PyTorch con el tutorial de CUDA en C.
* Subir a GitHub el notebook descargado y un reporte en Markdown con pantallazos y respuestas.

### Preguntas
1. Ahora que completaron todo el taller, ¿en qué se parece PyTorch a programar en CUDA directamente y en qué se diferencia? ¿Cuándo usarían uno y cuándo el otro?
2. Diagramen en Excalidraw el flujo completo del taller: desde la activación de la GPU hasta la predicción final. Úsenlo como resumen visual de todo lo que hicieron.
3. Si tuvieran que explicarle este taller a alguien que nunca ha programado, ¿cómo describirían en una sola analogía lo que hace una red neuronal entrenándose en una GPU?
