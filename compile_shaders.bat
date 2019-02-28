@ECHO OFF
SET AssetPath=%~dp0Assets

glslangValidator -V %AssetPath%\Shaders\shader.vert -o %AssetPath%\Shaders\vert.spv
glslangValidator -V %AssetPath%\Shaders\shader.frag -o %AssetPath%\Shaders\frag.spv
