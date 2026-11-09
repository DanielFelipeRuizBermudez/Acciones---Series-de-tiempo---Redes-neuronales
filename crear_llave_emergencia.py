"""Crea la llave de emergencia (break-glass) de la aplicación. Se corre una sola vez, en el equipo del administrador.

    python crear_llave_emergencia.py --salida "D:\\llave_emergencia.pem"

- Pide una frase dos veces y guarda la llave privada de emergencia cifrada con esa frase en el archivo de salida. El
  archivo y la frase nunca se suben al repositorio ni a Streamlit: se guardan aparte (por ejemplo, en una USB) y solo
  se usan para romper el vidrio cuando un usuario recupera su cuenta.
- Muestra la llave pública para pegarla en los Secrets de Streamlit como LLAVE_EMERGENCIA_PUBLICA.
"""

import argparse
import getpass
import sys
from pathlib import Path

import llaves

FRASE_MINIMA = 16


def main():
    parser = argparse.ArgumentParser(description="Crea la llave de emergencia de la aplicación.")
    parser.add_argument("--salida", required=True, help="archivo .pem fuera de la carpeta del proyecto")
    args = parser.parse_args()
    ruta = Path(args.salida).expanduser().resolve()
    proyecto = Path(__file__).resolve().parent
    if proyecto == ruta.parent or proyecto in ruta.parents:
        sys.exit("Guarda la llave fuera de la carpeta del proyecto, para que nunca termine en el repositorio.")
    if ruta.exists():
        sys.exit(f"Ya existe {ruta}; no se sobrescribe.")
    frase = getpass.getpass(f"Frase para proteger la llave (al menos {FRASE_MINIMA} caracteres): ")
    if len(frase) < FRASE_MINIMA:
        sys.exit(f"La frase debe tener al menos {FRASE_MINIMA} caracteres.")
    if getpass.getpass("Repite la frase: ") != frase:
        sys.exit("Las frases no coinciden.")
    privada, publica = llaves.crear_llave_emergencia(frase)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(privada)
    print(f"\nLlave privada de emergencia guardada en: {ruta}")
    print("Guárdala junto con la frase en un lugar seguro, fuera del repositorio y de Streamlit.\n")
    print("Pega esto en los Secrets de Streamlit (y en .streamlit/secrets.toml para probar en el equipo):\n")
    print('LLAVE_EMERGENCIA_PUBLICA = """')
    print(publica.strip())
    print('"""\n')
    print(f"Huella de la llave pública: {llaves.huella(publica)}")


if __name__ == "__main__":
    main()
