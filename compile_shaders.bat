@ECHO OFF
SET AssetPath=%~dp0Assets

glslangValidator -S vert -V %AssetPath%\Shaders\shader-vert.glsl -o %AssetPath%\Shaders\vert.spv
glslangValidator -S frag -V %AssetPath%\Shaders\shader-frag.glsl -o %AssetPath%\Shaders\frag.spv
