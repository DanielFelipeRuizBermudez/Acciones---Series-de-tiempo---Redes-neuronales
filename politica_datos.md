Versión del {version}. Esta política regula los datos que la aplicación {nombre_app} recoge al registrar a un usuario. Sigue la Ley 1581 de 2012 y el Decreto 1377 de 2013, compilado en la sección 2.2.2.25 del Decreto 1074 de 2015, y los lineamientos de la Circular Externa 002 de 2024 de la Superintendencia de Industria y Comercio sobre datos personales en sistemas de inteligencia artificial.

**1. Responsable del tratamiento**

- **Responsable:** {responsable}, estudiante de la Escuela Colombiana de Ingeniería Julio Garavito y autor del proyecto {nombre_app}.
- **Domicilio:** Bogotá D. C., Colombia.
- **Correo de contacto:** {correo}.
- **Responsable de atender consultas y reclamos:** {responsable}.

**2. Datos que se recogen**

| Dato | Obligatorio | Para qué se usa | Cómo se guarda |
|---|---|---|---|
| Alias | Sí | Identificar al usuario dentro de la aplicación. No tiene que ser el nombre real | En claro |
| Correo electrónico | Sí | Iniciar sesión y, solo si el usuario lo autoriza, ser contactado | Cifrado con la llave del usuario; para iniciar sesión se usa un índice HMAC de una sola vía |
| Nombre completo, teléfono, dirección y ciudad | Sí | Poder contactar al usuario, solo si él lo autoriza en cada caso | Cifrados con la llave del usuario |
| Fecha de nacimiento | Sí | Comprobar que el usuario es mayor de 18 años | Cifrada con la llave del usuario |
| Contraseña | Sí | Proteger la cuenta y la llave privada del usuario | Solo como hash bcrypt, nunca en texto plano |
| Par de llaves del usuario | Sí | Cifrar los datos personales para que solo el usuario los pueda leer | La llave pública en claro; la llave privada cifrada con la contraseña del usuario |
| Fecha y hora de la autorización y versión de la política aceptada | Sí | Probar que el titular autorizó el tratamiento | En claro |
| Efectivo ficticio disponible y fecha de inicio del portafolio | Sí | Llevar el portafolio simulado | En claro, asociado a un token aleatorio |
| Operaciones simuladas: fecha y hora, acción, compra o venta, cantidad, precio, comisión y consenso de los modelos en ese momento | Sí | Calcular las acciones y el saldo del usuario y compararlos con comprar y mantener | En claro, asociadas a un token aleatorio |
| Fecha del último acceso e intentos fallidos de contraseña | Sí | Detectar y bloquear accesos indebidos | En claro |
| Número de preguntas al agente por día | Sí | Aplicar la cuota diaria de {cuota} preguntas. Solo se guarda el número, no el texto | En claro, asociado a un token aleatorio |
| Preferencia sobre el historial de preguntas al agente | Sí | Saber si se guardan o no las conversaciones. Viene desactivada | En claro |
| Historial de preguntas al agente, ya anonimizadas, y sus respuestas | No | Mostrar las conversaciones anteriores. Solo se guarda si el usuario lo activa | Anonimizado, asociado a un token aleatorio |
| Solicitudes de acceso a los datos personales: quién las hizo, motivo, datos pedidos y autorizados, plazo, estado y fecha de cada consulta | Sí | Que el usuario controle y vea quién accedió a sus datos | En claro; la copia compartida, cifrada con la llave del solicitante y borrada al vencer |

La aplicación no recoge datos sensibles (salud, origen étnico, creencias, orientación política, datos biométricos), ni cédula, ni datos financieros o de cuentas de inversión reales. Es solo para mayores de 18 años: la fecha de nacimiento se usa para comprobarlo al registrarse.

**3. Finalidades**

1. Crear y administrar la cuenta del usuario.
2. Llevar su portafolio simulado: registrar sus operaciones ficticias, calcular su saldo y mostrarle cómo le fue frente a comprar y mantener y frente al consenso de los modelos.
3. Asignarle una cuota diaria de {cuota} preguntas al agente y responder sus preguntas. El texto de cada pregunta se anonimiza antes de enviarse a Google Gemini por medio de LiteLLM.
4. Contactar al usuario cuando sea necesario (por ejemplo, por un problema con su cuenta o por un cambio en esta política), solo con los datos que él decida compartir en cada caso, según la sección 5.
5. Proteger la seguridad de la aplicación y de las cuentas.
6. Elaborar estadísticas de uso agregadas y anonimizadas para el informe académico del proyecto.

Los datos no se venden, no se usan para publicidad y no se usan para perfilar al usuario como inversionista. No hay ranking público de usuarios.

**4. El usuario gobierna sus datos personales**

- Al registrarse, la aplicación crea para el usuario un par de llaves RSA de 3072 bits. La llave pública cifra; la llave privada descifra.
- La llave privada se guarda cifrada con una llave que se deriva de la contraseña del usuario (PBKDF2-HMAC-SHA256, 600.000 iteraciones). Como la aplicación no guarda la contraseña, sino solo su hash, nadie más puede abrir esa llave: ni el responsable, ni el administrador, ni quien tenga acceso a la base de datos.
- El nombre, el correo, el teléfono, la dirección, la ciudad y la fecha de nacimiento se guardan en un sobre cifrado con la llave pública del usuario (AES-256-GCM para los datos y RSA-OAEP para la llave de los datos). Solo el usuario, con su contraseña, puede verlos en la pestaña Mis datos.
- Si el usuario olvida su contraseña, sus datos personales no se pueden recuperar: tendría que crear una cuenta nueva.

**5. Acceso a los datos personales con autorización del titular**

1. Si el administrador necesita contactar a un usuario, le envía una solicitud desde la aplicación con un motivo, los datos que necesita y el tiempo que los necesita: 15 minutos, 1 hora o 24 horas.
2. El usuario ve la solicitud al iniciar sesión y decide. Puede rechazarla, o autorizarla con su contraseña y elegir qué datos comparte y por cuánto tiempo.
3. Al autorizar, la aplicación abre los datos con la llave privada del usuario y crea una copia, solo con los datos elegidos, cifrada con la llave pública del administrador. Solo el administrador, con su propia contraseña, puede abrir esa copia.
4. Cada vez que el administrador abre la copia, queda registrado, y el usuario lo ve en Mis datos.
5. El usuario puede revocar la autorización en cualquier momento. Al vencer el plazo, o al revocarla, la copia se borra.

**6. Encargados y transferencia internacional**

- **Streamlit Community Cloud** aloja la aplicación y **Neon** aloja la base de usuarios. Sus servidores están fuera de Colombia; por eso la autorización del registro incluye el consentimiento expreso para ese almacenamiento, como permite el artículo 26 de la Ley 1581. Los datos personales llegan a esos servidores ya cifrados con la llave del usuario.
- **Google (API de Gemini)** recibe solo el texto de las preguntas, ya anonimizado: los nombres, documentos, correos, teléfonos, números de cuenta, direcciones y fechas de nacimiento se reemplazan por etiquetas como [NOMBRE] antes de enviarse. La pregunta viaja por medio de LiteLLM, una librería que corre dentro de la aplicación. Google no recibe el alias, los datos personales ni la contraseña.

**7. Derechos del titular**

Según el artículo 8 de la Ley 1581, el usuario puede:

1. Conocer, actualizar y rectificar sus datos.
2. Pedir prueba de la autorización que otorgó.
3. Ser informado sobre el uso que se ha dado a sus datos, incluido quién los consultó y cuándo.
4. Presentar quejas ante la Superintendencia de Industria y Comercio.
5. Revocar la autorización o pedir que se supriman sus datos.
6. Acceder gratis a sus datos.

**8. Procedimiento para consultas y reclamos**

Desde la pestaña **Mis datos** el usuario puede ver todo lo que se guarda sobre él; abrir, descargar y corregir sus datos personales con su contraseña; responder y revocar solicitudes de acceso y ver su historial; cambiar su contraseña; activar, desactivar o borrar el historial del agente; reiniciar su portafolio y revocar la autorización eliminando la cuenta. También puede escribir a {correo}.

| Tipo de solicitud | Plazo de respuesta | Prórroga máxima |
|---|---|---|
| Consulta (conocer sus datos o su uso) | 10 días hábiles | 5 días hábiles |
| Reclamo (corregir, actualizar, suprimir o revocar) | 15 días hábiles | 8 días hábiles |

Si un reclamo llega incompleto, el responsable le pide al usuario lo que falta dentro de los 5 días siguientes. Si no lo completa en 2 meses, se entiende que desistió. Antes de acudir a la Superintendencia de Industria y Comercio, el usuario debe haber agotado este trámite ante el responsable.

**9. Seguridad de la información**

Las contraseñas se guardan con hash bcrypt; los datos personales se cifran con la llave pública del usuario y su llave privada solo se abre con su contraseña; la cuenta se bloquea temporalmente después de 5 intentos fallidos de contraseña; la sesión se cierra sola después de 30 minutos sin actividad; los datos personales abiertos en pantalla se ocultan solos a los 5 minutos; el portafolio, las operaciones y el uso del agente se asocian a un token aleatorio y no al correo; cada operación se valida en el servidor; las preguntas al agente se anonimizan antes de enviarse y se bloquean los intentos de manipularlo; el registro de auditoría del agente no guarda el texto ni el usuario; la versión publicada usa HTTPS, y las claves de la aplicación se guardan fuera del repositorio de código. Si se vulnera la seguridad de los datos, el responsable lo informará a la Superintendencia de Industria y Comercio y a los usuarios afectados.

**10. Conservación y vigencia**

- Los datos se conservan mientras la cuenta esté activa. Cuando el usuario elimina su cuenta desde Mis datos, sus datos, sus llaves y las solicitudes de acceso se borran de inmediato. El historial del agente también se puede borrar por separado.
- Las copias compartidas con el administrador se borran al vencer el plazo autorizado o cuando el usuario revoca el permiso.
- La base de usuarios estará vigente hasta el 31 de diciembre de 2026, cuando termina el proyecto académico. Después se elimina.
- Esta política rige desde el {version}. Las cuentas creadas con una versión anterior deben aceptar esta versión y completar sus datos para seguir usando el portafolio. Cualquier cambio sustancial futuro se comunicará en la aplicación antes de aplicarse.