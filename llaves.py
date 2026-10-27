"""Llaves asimétricas por usuario: solo el titular puede leer sus datos personales.

- Cada usuario tiene un par de llaves RSA de 3072 bits. La llave pública cifra y la privada descifra.
- La llave privada se guarda cifrada con AES-256-GCM, con una llave que se deriva de la contraseña del usuario
  (PBKDF2-HMAC-SHA256, 600.000 iteraciones y una sal aleatoria). Sin la contraseña nadie la puede abrir: ni el
  administrador, ni quien tenga acceso a la base de datos, ni CLAVE_DATOS.
- Los datos se guardan en un sobre híbrido: se cifran con una llave AES-256-GCM aleatoria, y esa llave se cifra con
  la llave pública RSA-OAEP del destinatario. Cifrar solo necesita la llave pública; leer necesita la privada.
- Cada sobre va atado a un contexto (el token del titular o el número de la solicitud): un sobre copiado a otra fila
  no se puede abrir.
- Para compartir datos, el titular abre su sobre con su llave privada y crea un sobre nuevo, solo con los campos que
  elige, cifrado con la llave pública de quien los pidió.
"""

import base64
import hashlib
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

BITS_RSA = 3072
ITERACIONES = 600_000
_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


class LlaveNoAbre(Exception):
    """La contraseña no abre la llave privada, o el sobre no corresponde a esta llave o a este contexto."""


def _b64(datos):
    return base64.b64encode(datos).decode("ascii")


def _de_b64(texto):
    return base64.b64decode(texto.encode("ascii"))


def _llave_de_contrasena(contrasena, sal, iteraciones):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=sal, iterations=iteraciones)
    return kdf.derive(contrasena.encode("utf-8"))


def proteger_privada(privada, contrasena, iteraciones=ITERACIONES):
    """Llave privada cifrada con la contraseña: {'privada_cifrada', 'sal', 'nonce', 'iteraciones'}."""
    sal, nonce = os.urandom(16), os.urandom(12)
    der = privada.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    cifrada = AESGCM(_llave_de_contrasena(contrasena, sal, iteraciones)).encrypt(nonce, der, b"llave-privada")
    return {"privada_cifrada": _b64(cifrada), "sal": _b64(sal), "nonce": _b64(nonce), "iteraciones": iteraciones}


def generar_par(contrasena):
    """Par nuevo: {'publica' (PEM), 'privada_cifrada', 'sal', 'nonce', 'iteraciones'}."""
    privada = rsa.generate_private_key(public_exponent=65537, key_size=BITS_RSA)
    publica = privada.public_key().public_bytes(serialization.Encoding.PEM,
                                                serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")
    return {"publica": publica, **proteger_privada(privada, contrasena)}


def abrir_privada(guardada, contrasena):
    """Llave privada lista para descifrar, o LlaveNoAbre si la contraseña no es la correcta."""
    llave = _llave_de_contrasena(contrasena, _de_b64(guardada["sal"]), int(guardada["iteraciones"]))
    try:
        der = AESGCM(llave).decrypt(_de_b64(guardada["nonce"]), _de_b64(guardada["privada_cifrada"]),
                                    b"llave-privada")
    except InvalidTag as error:
        raise LlaveNoAbre("La contraseña no abre la llave privada.") from error
    return serialization.load_der_private_key(der, password=None)


def cifrar_para(publica_pem, datos, contexto):
    """Sobre híbrido (texto JSON) con `datos` que solo puede abrir la llave privada de `publica_pem`."""
    llave_datos, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
    publica = serialization.load_pem_public_key(publica_pem.encode("ascii"))
    contenido = json.dumps(datos, ensure_ascii=False).encode("utf-8")
    return json.dumps({
        "llave_envuelta": _b64(publica.encrypt(llave_datos, _OAEP)),
        "nonce": _b64(nonce),
        "cifrado": _b64(AESGCM(llave_datos).encrypt(nonce, contenido, contexto.encode("utf-8"))),
    })


def abrir_sobre(privada, sobre, contexto):
    """Datos del sobre, o LlaveNoAbre si el sobre no es para esta llave o para este contexto."""
    partes = json.loads(sobre)
    try:
        llave_datos = privada.decrypt(_de_b64(partes["llave_envuelta"]), _OAEP)
        contenido = AESGCM(llave_datos).decrypt(_de_b64(partes["nonce"]), _de_b64(partes["cifrado"]),
                                                contexto.encode("utf-8"))
    except (InvalidTag, ValueError) as error:
        raise LlaveNoAbre("Este sobre no se puede abrir con esta llave.") from error
    return json.loads(contenido.decode("utf-8"))


def huella(publica_pem):
    """Huella SHA-256 corta de la llave pública, para reconocerla a simple vista."""
    resumen = hashlib.sha256(publica_pem.encode("ascii")).hexdigest().upper()
    return ":".join(resumen[i:i + 4] for i in range(0, 32, 4))
