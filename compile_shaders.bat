@ECHO OFF
SET AssetPath=%~dp0Assets

glslangValidator -V %AssetPath%\Shaders\shader.vert
glslangValidator -V %AssetPath%\Shaders\shader.frag