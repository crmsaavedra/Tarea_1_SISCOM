# Guía de uso de SisCom Morse.exe

## 1. Descargar y abrir la aplicación

1. En GitHub, abre **Releases** y descarga `SisCom Morse.exe`.
2. Guarda el archivo en una carpeta de tu equipo, por ejemplo `Documentos\SisCom Morse`.
3. En Windows 10/11 de 64 bits, haz doble clic en `SisCom Morse.exe`. Se abrirá la ventana **Laboratorio óptico Morse · Sistemas de Comunicaciones**.
4. No necesitas instalar Python, abrir Arduino IDE, iniciar un servidor ni usar el navegador para trabajar en la aplicación.

La app procesa los videos localmente. Los proyectos se guardan en `%LOCALAPPDATA%\SisCom Morse\projects`; si cambias el EXE de carpeta, tus proyectos guardados se mantienen.

## 2. Revisar un video con HOLA

Para probar un solo video, sin preparar todavía todo el proyecto:

1. Abre la pestaña **Prueba rápida**.
2. Pulsa **Seleccionar video** y elige la grabación que contiene el mensaje HOLA.
3. Pulsa **Analizar video** y espera a que finalice el procesamiento.
4. Revisa el texto decodificado, la comparación con HOLA y el CER. En las pestañas de resultados también puedes consultar la señal y el espectro, si el video permite calcularlos.

La prueba rápida estima ROI, umbral y duración T. Sirve para comprobar el video y el flujo de decodificación; **no sustituye** la calibración formal ni las cuatro mediciones de la tarea. La aplicación muestra la evidencia que sigue faltando.

## 3. Preparar el proyecto completo

En **Proyecto**, pulsa **Crear proyecto** y completa los datos solicitados. Ten listos estos archivos:

- Un video corto de calibración que muestre puntos, rayas y pausas.
- Cuatro videos de transmisión completos, uno por distancia: 0,5 m, 1,0 m, 2,0 m y 3,0 m.
- Una fotografía del montaje. El video demostrativo es opcional y debe durar como máximo 120 segundos.

También ingresa el mensaje esperado, nombres y aportes de ambos integrantes, fuente y control de la transmisión, materiales, iluminación, brillo, referencias y declaración de uso de IA. Al crear el proyecto, la app copia los videos y archivos seleccionados a su carpeta de proyecto.

## 4. Calibrar, decodificar y analizar

1. **Calibración:** usa el video corto de calibración. Revisa la vista previa, arrastra sobre el LED para ajustar la ROI y define el umbral, T y tolerancia. Pulsa **Analizar y guardar calibración**. T es la duración de un punto; una raya dura 3T y las pausas entre símbolos, letras y palabras duran T, 3T y 7T.
2. **Decodificación:** selecciona cada distancia, revisa su ROI y estado de adquisición y procesa las grabaciones con la calibración guardada.
3. **Análisis:** consulta la tabla de errores y los gráficos por distancia, incluyendo CER, FFT, PSD/B95 y la comparación entre 0,5 m y 3,0 m.
4. **Exportación:** revisa la lista de requisitos pendientes. Cuando la app habilite la entrega, guarda el ZIP de resultados y verifica su contenido antes de preparar el informe PDF de la tarea.

Una señal débil o un texto recibido vacío no implica automáticamente que la adquisición sea inválida. Márcala inválida solo si la grabación quedó incompleta por un fallo de adquisición e indica el motivo.

## 5. Consejos para una grabación legible

- Mantén el LED visible, enfocado y dentro del cuadro durante toda la transmisión.
- Evita reflejos, luces que parpadeen y movimientos de la cámara.
- Incluye pausas suficientes antes y después del mensaje para distinguir el inicio y el final.
- Reproduce la calibración y las cuatro distancias con el mismo formato de mensaje y duración de punto.
- Si no se detectan pulsos, revisa el video, la orientación y la ROI antes de interpretar el CER.

## 6. Ejecutar desde el código fuente (opcional)

Para usar solo la aplicación, abre el EXE. Si necesitas reconstruirla desde el código en Windows, instala Python 3.11 o 3.12 de 64 bits con Tcl/Tk y ejecuta `build_exe.bat`. El resultado se crea como `dist\SisCom Morse.exe`.

El sketch `arduino_morse_leds.ino` es independiente de la interfaz: se carga en Arduino IDE para transmitir Morse con los LED conectados a D2–D9.
