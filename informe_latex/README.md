# Informe LaTeX

- `main.tex`: fuente editable del informe.
- `Informe_Tarea1_SISCOM.pdf`: PDF compilado y revisado.
- `images/udec_.jpg`: logo tomado del formato de informe indicado por el grupo.

## Completar el borrador

Los textos naranjos entre corchetes son campos que debe completar el grupo con sus datos. Los recuadros grises son espacios para las figuras; cada uno indica un nombre de archivo sugerido y va seguido por un recuadro de interpretación. Copie las figuras a esta carpeta con el nombre indicado y reemplace el recuadro correspondiente por `\includegraphics`.

El video preliminar disponible contiene HOLA. El informe lo separa explícitamente de la evidencia evaluada: hace falta registrar la calibración a 0,5 m y las cuatro transmisiones SIS COM a 0,5, 1,0, 2,0 y 3,0 m. No complete resultados experimentales hasta procesar esas grabaciones.

## Compilar

Con una distribución LaTeX que incluya `latexmk`:

```powershell
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
```

El comando deja el PDF de trabajo y sus archivos auxiliares en `build/`. El PDF revisado que se entrega está en `Informe_Tarea1_SISCOM.pdf`.
