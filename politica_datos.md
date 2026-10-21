Versión del {version}. Esta política regula los datos que la aplicación {nombre_app} recoge al registrar a un usuario. Sigue la Ley 1581 de 2012 y el Decreto 1377 de 2013, compilado en la sección 2.2.2.25 del Decreto 1074 de 2015.

**1. Responsable del tratamiento**

- **Responsable:** equipo del proyecto {nombre_app}, estudiantes de la Escuela Colombiana de Ingeniería Julio Garavito.
- **Domicilio:** Bogotá D. C., Colombia.
- **Correo de contacto:** {correo}.
- **Responsable de atender consultas y reclamos:** {responsable}.

**2. Datos que se recogen**

| Dato | Obligatorio | Para qué se usa |
|---|---|---|
| Alias | Sí | Identificar al usuario dentro de la aplicación. No tiene que ser el nombre real |
| Correo electrónico | Sí | Iniciar sesión y comunicar cambios de la cuenta o de esta política. Se guarda cifrado y aparte del resto de los datos |
| Contraseña | Sí | Proteger la cuenta. Se guarda solo como hash (bcrypt), nunca en texto plano |
| Confirmación de mayoría de edad, fecha y hora de la autorización y versión de la política aceptada | Sí | Probar que el titular autorizó el tratamiento |
| Efectivo ficticio disponible y fecha de inicio del portafolio | Sí | Llevar el portafolio simulado |
| Operaciones simuladas: fecha y hora, acción, compra o venta, cantidad, precio, comisión y consenso de los modelos en ese momento | Sí | Calcular las acciones y el saldo del usuario y compararlos con comprar y mantener |
| Fecha del último acceso e intentos fallidos de inicio de sesión | Sí | Detectar y bloquear accesos indebidos |
| Número de preguntas al agente por día | Sí | Aplicar la cuota diaria de {cuota} preguntas. Solo se guarda el número, no el texto |
| Preferencia sobre el historial de preguntas al agente | Sí | Saber si se guardan o no las conversaciones. Viene desactivada |
| Historial de preguntas al agente, ya anonimizadas, y sus respuestas | No | Mostrar las conversaciones anteriores. Solo se guarda si el usuario lo activa |

La aplicación no recoge datos sensibles (salud, origen étnico, creencias, orientación política, datos biométricos), ni cédula, ni datos financieros o de cuentas de inversión reales. Es solo para mayores de 18 años: si se detecta la cuenta de un menor, se elimina.

**3. Finalidades**

1. Crear y administrar la cuenta del usuario.
2. Llevar su portafolio simulado: registrar sus operaciones ficticias, calcular su saldo y mostrarle cómo le fue frente a comprar y mantener y frente al consenso de los modelos.
3. Asignarle una cuota diaria de {cuota} preguntas al agente y responder sus preguntas. El texto de cada pregunta se anonimiza antes de enviarse a Google Gemini por medio de LiteLLM.
4. Proteger la seguridad de la aplicación y de las cuentas.
5. Elaborar estadísticas de uso agregadas y anonimizadas para el informe académico del proyecto.

Los datos no se venden, no se usan para publicidad y no se usan para perfilar al usuario como inversionista. No hay ranking público de usuarios.

**4. Encargados y transferencia internacional**

- **Streamlit Community Cloud** aloja la aplicación y **Neon** aloja la base de usuarios. Sus servidores están fuera de Colombia; por eso la autorización del registro incluye el consentimiento expreso para ese almacenamiento, como permite el artículo 26 de la Ley 1581.
- **Google (API de Gemini)** recibe solo el texto de las preguntas, ya anonimizado: los nombres, documentos, correos, teléfonos y números de cuenta se reemplazan por etiquetas como [NOMBRE] antes de enviarse. La pregunta viaja por medio de LiteLLM, una librería que corre dentro de la aplicación. Google no recibe el alias, el correo ni la contraseña.

**5. Derechos del titular**

Según el artículo 8 de la Ley 1581, el usuario puede:

1. Conocer, actualizar y rectificar sus datos.
2. Pedir prueba de la autorización que otorgó.
3. Ser informado sobre el uso que se ha dado a sus datos.
4. Presentar quejas ante la Superintendencia de Industria y Comercio.
5. Revocar la autorización o pedir que se supriman sus datos.
6. Acceder gratis a sus datos.

**6. Procedimiento para consultas y reclamos**

Desde la pestaña **Mis datos** el usuario puede ver todo lo que se guarda sobre él, descargarlo, corregir su alias y su correo, cambiar su contraseña, activar, desactivar o borrar el historial del agente, reiniciar su portafolio y revocar la autorización eliminando la cuenta. También puede escribir a {correo}.

| Tipo de solicitud | Plazo de respuesta | Prórroga máxima |
|---|---|---|
| Consulta (conocer sus datos o su uso) | 10 días hábiles | 5 días hábiles |
| Reclamo (corregir, actualizar, suprimir o revocar) | 15 días hábiles | 8 días hábiles |

Si un reclamo llega incompleto, el equipo le pide al usuario lo que falta dentro de los 5 días siguientes. Si no lo completa en 2 meses, se entiende que desistió. Antes de acudir a la Superintendencia de Industria y Comercio, el usuario debe haber agotado este trámite ante el equipo.

**7. Seguridad de la información**

Las contraseñas se guardan con hash bcrypt; la cuenta se bloquea temporalmente después de 5 intentos fallidos de inicio de sesión; la sesión se cierra sola después de 30 minutos sin actividad; el correo se guarda cifrado; el portafolio, las operaciones y el uso del agente se asocian a un token aleatorio y no al correo; cada operación se valida en el servidor; las preguntas al agente se anonimizan antes de enviarse y se bloquean los intentos de manipularlo; el registro de auditoría del agente no guarda el texto ni el usuario; la versión publicada usa HTTPS, y las claves de la aplicación se guardan fuera del repositorio de código. Si se vulnera la seguridad de los datos, el equipo lo informará a la Superintendencia de Industria y Comercio y a los usuarios afectados.

**8. Conservación y vigencia**

- Los datos se conservan mientras la cuenta esté activa. Cuando el usuario elimina su cuenta desde Mis datos, sus datos se borran de inmediato. El historial del agente también se puede borrar por separado.
- La base de usuarios estará vigente hasta el 31 de diciembre de 2026, cuando termina el proyecto académico. Después se elimina.
- Esta política rige desde el {version}. Cualquier cambio sustancial se comunicará por correo antes de aplicarse.
