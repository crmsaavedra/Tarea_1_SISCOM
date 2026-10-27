# Laboratorio óptico Morse

Aplicación local de escritorio para organizar las grabaciones de una transmisión Morse óptica, calibrar el receptor, decodificar señales, analizar resultados y preparar un paquete reproducible.

**Guía para abrir y usar el ejecutable:** [GUIA_USO_EXE.md](GUIA_USO_EXE.md). Descarga `SisCom-Morse.exe` desde la sección **Releases** del repositorio.

## Informe de la tarea

El informe editable en LaTeX y su PDF compilado están en [`informe_latex/`](informe_latex/). Además del sistema y el procedimiento, el marco teórico desarrolla el enlace OOK, la propagación óptica, el muestreo y exposición de la cámara, la ROI, el umbral, la tolerancia temporal y la interpretación de FFT, PSD y B95, con referencias técnicas. Incluye espacios rotulados para las figuras y recuadros separados para escribir sus interpretaciones. Antes de entregarlo, hay que completar los campos naranjos con mediciones reales e insertar las figuras solicitadas; el clip preliminar HOLA no reemplaza las grabaciones SIS COM de las cuatro distancias.

## Ejecutar la aplicación

La versión entregable es un solo archivo Windows: `dist\SisCom Morse.exe`. Haz doble clic y se abrirá una ventana Tkinter; no inicia un navegador, servidor web ni necesita Internet para procesar videos.

Los proyectos se guardan bajo `%LOCALAPPDATA%\SisCom Morse\projects`. El procesamiento se hace en el equipo y los videos no se suben a ningún servicio.

## Crear el EXE

Para construirlo desde el código se necesita Windows 10/11, Python 3.11 o 3.12 de 64 bits con Tcl/Tk incluido y conexión a Internet durante la instalación inicial de dependencias. Ejecuta `build_exe.bat`; instala los paquetes, compila en modo `--onefile --windowed` y deja el ejecutable en `dist\SisCom Morse.exe`.

`run_app.bat` es solo un lanzador opcional para desarrollo desde el código fuente. Para usar la app normalmente, ejecuta el EXE.

## Preparar el proyecto de la tarea

En **Proyecto**, carga:

- Un video breve de calibración con puntos, rayas y pausas.
- Cuatro transmisiones completas SIS COM, una por distancia: 0,5; 1,0; 2,0 y 3,0 m.
- Los nombres y aportes de ambos integrantes, y los datos de la fuente, el control, materiales, iluminación y brillo.
- Las referencias consultadas y la declaración de uso de IA.
- Una foto del montaje. El video demostrativo es opcional y debe durar como máximo 120 segundos.

La app copia esos archivos dentro del proyecto y conserva rutas relativas en `project.json`.

## Prueba rápida

La sección **Prueba rápida** analiza un único video con HOLA. Estima la región de interés (ROI), el umbral y la duración T, decodifica desde la señal y luego compara el resultado con HOLA para mostrar sustituciones, eliminaciones, inserciones y CER. También presenta la señal temporal y FFT/PSD/B95 cuando los datos permiten calcularlos.

Esta prueba no reemplaza la calibración ni las cuatro mediciones de la tarea. La interfaz enumera la evidencia que todavía falta.

## Flujo de trabajo

La aplicación tiene seis secciones: **Proyecto**, **Prueba rápida**, **Calibración**, **Decodificación**, **Análisis** y **Exportación**. La ventana se organiza en vista de video y ROI a la izquierda, controles al centro y pestañas de resultado, señal y espectro a la derecha. Los clips horizontales se giran solo para la vista previa; el video original y las coordenadas de ROI no se modifican.

### Calibración y decodificación

La calibración utiliza el video breve. Define una ROI, un umbral de intensidad digital, la duración T de un punto y una tolerancia. La raya dura 3T y las pausas entre símbolos, letras y palabras duran T, 3T y 7T. La calibración se reutiliza para las cuatro distancias.

El texto esperado se utiliza solo después de decodificar para evaluar CER. Una señal débil o texto recibido vacío sigue siendo una adquisición válida; márcala inválida solo si la grabación está incompleta por un fallo de adquisición, indicando el motivo.

### Análisis

La tabla por distancia incluye duración, sustituciones, eliminaciones, inserciones, CER, texto recibido y estado de adquisición. La app guarda CSV, textos, decisiones, notas y gráficos de CER, señal, FFT, PSD/B95 y comparación de 0,5 m frente a 3,0 m. Los segmentos de comparación se alinean con tiempos de inicio editables.

La FFT es unilateral. La PSD usa Welch y ventana Hann; B95 corresponde a la frecuencia que integra el 95 % de la potencia estimada. Los resultados guardados se vinculan a la calibración, ROI, videos y tiempos usados, y se marcan obsoletos cuando cambia esa configuración.

### Exportación

La sección de entrega muestra los requisitos pendientes y solo habilita la creación del ZIP cuando el análisis y la evidencia verificable están completos. El paquete incluye los videos, la configuración, el código fuente y resultados. No redacta el informe PDF ni reemplaza la demostración oral.

## Transmisor Arduino

El sketch `arduino_morse_leds.ino` controla LED conectados a D2–D9. Usa una resistencia individual de 220–330 Ω en serie con cada LED. Abre el Monitor Serie a 9600 baudios, selecciona `Nueva línea` y envía letras A–Z, dígitos 0–9 y espacios. La unidad Morse T es 200 ms por defecto.

## Pruebas de desarrollo

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
