# Despliegue de VibroVision en AWS

El proyecto usa la nueva experiencia de AWS, el perfil local `vibrovision` y la región seleccionada `us-east-2`. La autenticación se hace con `aws login`, sin claves estáticas dentro del código. Agent Toolkit se conecta a su servicio en `us-east-1`; eso no cambia la región de los recursos de la aplicación.

## Puesta en marcha verificada — 26 de octubre de 2026 (México)

URL de la demo activa: **https://d30icy4clt3p0c.cloudfront.net/**. Los stacks `vibrovision` y `vibrovision-releases` terminaron en `CREATE_COMPLETE` en `us-east-2`.

La clave está únicamente en `.aws-deploy/vibrovision-access.json` dentro de esta carpeta del proyecto, con permisos `0600`. Copiar el campo `api_key` al campo **API key** de la página y pulsar **Compare the two real fan videos** para ejecutar una comparación nueva. La URL se puede compartir; para ejecutar análisis también hace falta la clave. El archivo completo contiene credenciales y debe mantenerse privado.

La prueba externa confirmó HTTPS, página web, OpenCV **5.0.0** en **aarch64**, rechazo `401` sin clave, escenarios autorizados, comparación nueva de ambos clips y descarga autenticada de la gráfica. El análisis tardó **45,5 s** en esta ejecución y produjo una razón de amplitudes de **35,90×**; es una medición de estos dos videos, no una promesa de sensibilidad ni un benchmark de rendimiento. El informe se verificó en S3 con cifrado `AES256` y bloqueo de acceso público. El resultado reproducible está en `.aws-deploy/verification.json` y la gráfica en `.aws-deploy/comparison-deployed.png`. Las **60 pruebas locales** pasan.

En el primer arranque se detectó que la extracción bajo `umask 077` había dejado la carpeta estática con permiso `0700`. Se corrigió a `0755` en la instancia, el contenedor activo y la imagen guardada, sin cambiar los archivos de la aplicación. Para siguientes despliegues, el empaquetador incluye permisos explícitos de directorios y la extracción los conserva; una prueba reproduce esa condición. El registro privado del despliegue distingue las plantillas originalmente ejecutadas de esta prevención posterior. La aplicación permanece encendida por autorización del usuario.

## Qué se publica

Una instancia Graviton `t4g.small` (dos vCPU, 2 GiB) ejecuta el contenedor con OpenCV 5. Se eligió después de consultar las instancias Arm aptas para el plan gratuito del proyecto. El modo de créditos de CPU es `standard`: puede reducir velocidad al agotar créditos, sin activar cargos por créditos excedentes. El búfer de fotogramas se limita a 512 MiB y el análisis sigue siendo secuencial.

CloudFront ofrece una URL HTTPS con certificado del servicio. El origen EC2 acepta solamente las direcciones administradas de CloudFront y comprueba una cabecera secreta. La API exige además la clave de acceso del usuario. La página de presentación y `/healthz` son públicas. Las respuestas dinámicas no se guardan en caché. Los parámetros de clave en las URLs de imágenes se ocultan en los registros de Uvicorn antes de enviarlos a CloudWatch.

La conexión navegador–CloudFront está cifrada. El tramo CloudFront–EC2 usa HTTP, limitado por la lista de prefijos y la cabecera de origen. Es una configuración de demostración; un despliegue de producción con video confidencial necesita cifrar también ese tramo y definir autenticación y aislamiento por usuario.

Se guardan dos grupos de datos en S3 privado, cifrados y accesibles por HTTPS: las versiones de código y la evidencia de análisis. La instancia puede leer únicamente su archivo de código y escribir/leer evidencia bajo `jobs/`. La clave se genera en Secrets Manager y la instancia la obtiene con su rol: su valor no aparece en la plantilla ni en los datos de arranque. Los volúmenes EBS están cifrados y el acceso de administración usa SSM, sin SSH.

CloudWatch conserva los registros durante 14 días y una alarma cuenta las escalaciones. Esta plantilla no suscribe personas a notificaciones. El historial de eventos de administración de CloudTrail ofrece la auditoría predeterminada de 90 días; en el proyecto nuevo no se encontró un trail persistente hacia S3. No se modificó la configuración de auditoría del proyecto.

## Publicar desde la carpeta local

Requiere AWS CLI v2 con sesión autorizada, Python 3 y permisos para los recursos descritos. No necesita un repositorio Git público ni Docker instalado en el equipo local.

```sh
aws login --region us-east-2 --profile vibrovision
python3 infra/deploy_aws.py plan --profile vibrovision --region us-east-2
```

`plan` empaqueta únicamente el código de ejecución, el Dockerfile, los requisitos y los dos clips de ejemplo. Excluye credenciales, notebooks y resultados de trabajo. Calcula SHA-256 y prepara dos conjuntos de cambios de CloudFormation. En ese punto se crean registros de revisión, sin aprovisionar EC2 ni la aplicación.

Después de revisar y autorizar los cambios:

```sh
python3 infra/deploy_aws.py execute --profile vibrovision --region us-east-2
python3 infra/deploy_aws.py verify --profile vibrovision --region us-east-2
```

El arranque descarga el archivo privado, verifica su SHA-256 y construye el contenedor nativo para Arm64. CloudFormation no declara la instancia lista hasta comprobar OpenCV 5, arquitectura `aarch64`, salud y huella de la versión. Después publica la distribución HTTPS.

`verify` comprueba el HTTPS, el rechazo de solicitudes sin clave, los escenarios autorizados, una comparación nueva de los videos reales y la persistencia cifrada de su informe en S3 privado. Guarda el resultado sin secretos en `.aws-deploy/verification.json`. La clave queda en `.aws-deploy/vibrovision-access.json`, con permisos `0600`; la carpeta está excluida de Git. No publicar ese archivo ni incluirlo en capturas.

El controlador está destinado al alta inicial. No ejecuta una actualización ni reemplaza una instancia existente de forma implícita. Si falla un paso, revisar los conjuntos de cambios y los eventos antes de reintentar; no borrar recursos automáticamente.

## Consultar y administrar

```sh
python3 infra/deploy_aws.py status --profile vibrovision --region us-east-2
aws cloudformation describe-stacks --stack-name vibrovision --profile vibrovision --region us-east-2
```

Usar el `InstanceId` de la salida para acceder por SSM. Las pruebas de rendimiento de Graviton/COOL requieren una evaluación específica; tener OpenCV en AWS no demuestra por sí mismo una aceleración ni precisión industrial.

Los recursos consumen los créditos del proyecto según el uso. Detener EC2 reduce el consumo de cómputo; siguen existiendo el volumen EBS, S3, Secrets Manager y la distribución. No se ha programado una eliminación automática.

Los buckets tienen retención al borrar el stack. La evidencia caduca a 30 días, también para versiones antiguas; las versiones de código se conservan. La eliminación completa requiere decidir qué evidencia y versiones conservar y vaciar únicamente los buckets de este despliegue. No borrar la VPC predeterminada ni recursos ajenos al proyecto.

## Validación reproducible

Las plantillas se verifican con `cfn-lint 1.58.0` para `us-east-2`. Las reglas de `infra/security.guard` proceden del registro oficial de AWS, commit `7f7340c26ae5d5e8874651dbffeb12e0e9f505b6`, y añaden comprobaciones para IMDSv2, cifrado de los volúmenes de la instancia y caché dinámica desactivada. El ejecutable oficial de CloudFormation Guard 3.2.1 se verificó contra su SHA-256 publicado. El enlace `guardpycfn 0.1.0` falló con Python 3.14 en macOS y no se utilizó como prueba de cumplimiento.

La validación de servicio se conserva en los archivos `*-validation.json` dentro de `.aws-deploy/`. Sintaxis local, reglas de seguridad, validación de AWS y pruebas reales del servicio son comprobaciones distintas.
