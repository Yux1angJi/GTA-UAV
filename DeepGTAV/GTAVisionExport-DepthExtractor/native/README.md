**GTAVisionExport (native) build steps**
-------------------------------------
Note: there may be some unessesary steps that could be removed from this procedure at some point, but this is what I did. Also I have a tendency write Linux style paths. All paths (except URLs) should be Windows style.

My set-up:
----------
GTX770, Windows 10 64bit, Visual Studio 2017 (community edition), cmake 3.9.0, GTAV version 1.0.231.0

Needed tools and libraries
--------------------------
AsiLoader and ScriptHookV	: http://www.dev-c.com/files/ScriptHookV_1.0.1180.2.zip

cmake				: https://cmake.org/download/

Eigen3				: http://bitbucket.org/eigen/eigen/get/3.3.1.tar.bz2

MS Visual Studio 2017 		: https://www.visualstudio.com/cs/downloads/

MSBuild.exe in `PATH` variable, version 141.


Build steps
-----------
1. git clone https://github.com/umautobots/GTAVisionExport (**latest!** as of today)
2. Extract Eigen3 somwhere convenient (Your GTAVisionExport folder is as good as any).
3. Extract ScriptHookV archive and drop the files in 'bin' into your GTAV exe folder.
4. Run cmake (cmake-gui) from your Windows start menu.
5. Hit 'Browse Source' and select your GTAVisionExport/native folder.
6. Hit 'Browse Build', create GTAVisionExport/native/build folder and select it.
7. Hit 'configure' (first time around it will fail but dont worry).
8. Choose project generator 'Visual Studio 15 2017 Win64' and keep the option 'use default native compilers'
9. After the fail dialog, modify the EIGEN3_INCLUDE_DIR to point to your Eigen3 folder.
10. Run 'configure' followed by 'generate'.
11. cmake should now have generated the Visual Studio solution into GTAVisionExport/build.
12. Open 'GTANativePlugin.sln' in Visual Studio.
13. Select 'release' from the 'Solution Configurations' drop down.
14. Edit GTAVisionNative project properties/configuration properties/c/c++/additional include dirs in VS to add the GTAVisionExport/native/src folder (this allows VS to find MinHook.h)
15. Edit GTAVisionNative project properties/configuration properties/linker/input/additional dependencies to add : 
`"..\..\deps\libMinHook.x64.lib"` 
16. Press F6 to build the solution. it should now succeed and the products should be in 'GTAVisionExport\native\build\src\Release'
17. Copy GTAVisionNative.asi & GTAVisionNative.lib to your GTAV exe folder.
18. Run GTAV.
19. Get to a place where you want to grab frames and press 'l' (lowercase 'L') to grab a frame. GTAVisionExport should now create color.raw, stencil.raw and depth.raw files in your GTAV exe folder.

HTH

FAQ
---

Can not configure in CMake, `gdi32.lib` is missing:
This is probably due to incorrect Visual Studio SDK, can be solved by installing Windows 10 SDK (10.0.15063.0) for Desktop C++ x86 and x64 in the VS Installer. 

Source: https://stackoverflow.com/questions/33599723/fatal-error-lnk1104-cannot-open-file-gdi32-lib

The game crashes after pressing 'L':

If you are using steam, be sure to disable the steam overlay for this game.

If steam overlay is disabled and game still crashes, be sure to have resolution same or higher as constants in source code of this project.
Default is 1000 x 1000, as can be seen [here](https://github.com/umautobots/GTAVisionExport/blob/4f2bf90997df056764605076c0c95b885c424c43/native/src/main.cpp#L235) and [here](https://github.com/umautobots/GTAVisionExport/blob/4f2bf90997df056764605076c0c95b885c424c43/native/src/main.cpp#L212).
If you want to use lower resolution, change these numbers and recompile.
