# Livery Maker

Makes X-Plane liveries for a whole fleet from one design: airline name, colours, titles, tail symbol, and a
registration and SELCAL for each aircraft. It runs entirely on your Mac, with no internet connection or Claude needed.

## Build the app (once)

1. Open **Terminal**, then type `cd ` (with a space) and drag this `LiveryMaker` folder into the window. Press Return.
2. Run:

       bash build_mac.sh

   The first build downloads the libraries it needs and takes a few minutes.
3. The app is at `dist/Livery Maker.app`. Drag it into **Applications** if you like.

You need Python 3.10 or newer. If the script says it can't find it, install Python from
<https://www.python.org/downloads/macos/> and run the script again.

To run from the source code instead of the app (handy after editing it): `bash run.sh`.

## Build the app on Windows (once)

Python is only needed to build the app. The finished app carries its own copy and runs without it.

1. Install Python 3.12 from <https://www.python.org/downloads/windows/>. On the first screen of the installer, tick
   **Add python.exe to PATH**, then click **Install Now**.
2. Copy this `LiveryMaker` folder to the PC, for example to `Documents\LiveryMaker`.
3. Double-click **`build_windows.bat`** in that folder. The first build downloads the libraries it needs and takes a
   few minutes; a black window shows progress and says *Done* at the end.
4. The app is in `dist\Livery Maker\`. Keep `Livery Maker.exe` together with the `_internal` folder next to it. You
   can move the whole `Livery Maker` folder anywhere, for example `C:\Program Files` or your Desktop. Right-click
   the .exe and choose **Send to → Desktop (create shortcut)** for a shortcut.

The first time you open it, Windows may show *Windows protected your PC*, because the app isn't signed by a known
publisher. Click **More info**, then **Run anyway**. It only asks once.

To run from the source code instead (handy after editing it): double-click `run_windows.bat`.

On the X-Plane PC you can point the app straight at the X-Plane folders: the aircraft folder for each 3D model, and the
aircraft's `liveries` folder for the base liveries. Choosing the real aircraft folder for the 3D preview also gives
the 3D view the aircraft's own textures for parts the livery doesn't repaint, such as wings and gear.

Projects store full folder paths, so a project made on the Mac shows red dots on the PC until you choose the folders
again on the Aircraft files tab (and the other way round).

## Using it

1. **File → Open project…** and choose `FlyFurball.lmproj` to start from the FlyFurball fleet, or start a new project.
2. **Aircraft files** tab: point each aircraft type at its paintkit or base livery folder (see below). The app only
   reads these files and never changes them. A green ✓ means the folder is ready.
3. **Design** tab: set the airline name, titles, fonts, colours, the fade towards the tail and the tail symbol
   (a saltire, your own PNG logo, or nothing).
4. **Fleet** tab: add one row per aircraft, with its registration, engine or version, and SELCAL.
5. Choose an output folder, then:
   - **Preview selected aircraft** builds just that aircraft and shows its textures.
   - **Build all liveries** writes one ready-to-install folder per aircraft, plus `<airline> fleet.xlsx`.
     Tick *Also make a .zip* to get zips for copying to the X-Plane PC.
6. Copy each livery folder into that aircraft's `liveries` folder on the X-Plane PC.

The first aircraft of each type takes longer, around 1–3 minutes, while its paintkit loads. After that, each extra
aircraft of the same type takes about 20–40 seconds.

## Around the window

- **Toolbar:** **Preview** (⌘R) previews the selected Fleet row, **Build all** (⌘B) builds every livery, **Stop** (⌘.)
  stops after the current aircraft. The same commands are in the **Build** menu.
- **Live preview:** with *Live preview* ticked, any change refreshes the last preview by itself a couple of seconds
  later, keeping your 3D view angle.
- **Design tab:** the strip at the top is a quick picture of your colours, titles and tail. Click a section heading to
  fold it away; the app remembers which are open.
- **Fleet tab:** the *Status* column shows *Ready*, *Built*, *Changed* (built, but the design has changed since),
  *Built earlier*, *No registration* or *Files missing*. Double-click a status to preview that aircraft.
- **Progress:** a slim bar shows how far Build all has got. Click *Details* for the full log.
- **Getting started:** until your first preview, the Preview area shows a short checklist with a button for each step.

## Moving the titles, registration and tail logo

The **Positions** tab (next to **Preview**) shows the left side of the fuselage as a flat strip, or the fin standing
upright. Choose the aircraft, choose what to move (**Title**, **Registration** or **Tail logo**), and press **Load side
view**; the app builds that aircraft once without them, which takes about a minute the first time. Then:

- drag the title to move it, or use the arrow keys (hold Shift for bigger steps);
- use the **Size** slider to make it bigger or smaller;
- press **Reset** to put it back where it started.

The title stays centred on the window line (the dashed orange line) unless you pull it clearly above or below, and the
registration stays at its usual height the same way. The right-hand side always follows, so you only place each once.
Each aircraft type keeps its own positions, saved in the project. Use **Preview selected aircraft** to see the result on the 3D model.

## 3D preview (all five aircraft)

**Preview selected aircraft** adds a **3D** tab that shows the livery on the aircraft's own model. To turn it on, set
the *3D preview* folder on the **Aircraft files** tab. It needs that aircraft's `.obj` files and its `.acf` file:
either the aircraft folder itself, or a copy of just those files. The app only reads them. Parts the livery doesn't
repaint (such as the Q4XP's wings, gear and props) show the aircraft's own textures if they're in that folder, and plain
colours if they aren't.

In the 3D tab, drag to turn the aircraft, scroll or pinch to zoom, right-drag (or Shift-drag) to move it, and
double-click to reset. The buttons jump to the left, right, front and top views. The aircraft is shown parked: gear
down, doors shut, and with the engine type set on the Fleet row.

## Supported aircraft and what to point them at

| Aircraft | Folder to choose | Notes |
|---|---|---|
| ToLiss A319 | A319 Paintkit V2.1 | Engines CFM or IAE, set through `livery.tlscfg`. |
| ToLiss A321 / A321neo | A321 Paintkit V2 | ceo CFM, ceo IAE, neo LEAP, neo PW. Classic four-door fuselage. |
| Embraer ERJ145 | The Loganair (G-SAJC) livery | No paintkit exists, so it repaints that livery. |
| FlyJSim Q4XP | The FlyJSim house livery (G-JEDI) | Writes DXT1 .dds files and `LiveryStat.prf` (registration and SELCAL). |
| X-Crafts E190 | The KLM Cityhopper (PH-EXD) livery | Writes DXT5 .dds files, the cockpit placard and `tail.txt`. Only the E190 variant is made. |

The ERJ145, Q4XP and E190 templates were measured on those exact base liveries. If you choose a different one, the app
warns you, because titles and markings may land in the wrong places.

## Good to know

- **Body colour:** the fuselage colour, which fades into the accent colour towards the tail. On the ERJ145, Q4XP and
  E190, small parts the old livery left white (doors, hatches, nose-gear bay) stay white.
- **Title outline:** on the Design tab, next to *Title outline*, pick a colour and drag the slider to set the
  thickness as a percentage of the letter height (0 = no outline). The outline sits outside the letters, so they keep
  their shape. It shows in the Title position view and on every aircraft.
- **Weathering:** the slider on the Design tab sets how dirty the ToLiss A319 and A321 look, using the dirt layers
  in their paintkits. The middle is the paintkit as supplied, far left is spotless, far right is heavily weathered. The
  ERJ145, Q4XP and E190 keep the weathering of the livery they are built on.
- **Tail logo:** choose the saltire, your own PNG (with a transparent background) or nothing on the Design tab. Its
  size there applies to every aircraft; the Positions tab moves it, fine-tunes the size and tilts it per aircraft
  (Angle: plus leans the top towards the nose, and the other side of the fin gets the matching tilt). It reads
  correctly on both sides of the fin.
- **Fonts:** the built-in fonts are Inter (SIL Open Font License, see `livery_maker/fonts/OFL.txt`). You can also
  choose any .ttf or .otf file for the titles.
- **Real airlines:** registrations, names and logos you add are your responsibility. Check a registration isn't
  already in use before you rely on it.
- **Without the window:** `"/Applications/Livery Maker.app/Contents/MacOS/Livery Maker" --build myfleet.lmproj`
  builds everything from the command line.

## For later: adding another aircraft

Each aircraft is one file in `livery_maker/engine/templates/`. It describes where things sit on that add-on's
textures: the fuselage bands, the window line, the fin outline and how the two fin sides map onto each other, and
the registration position. Copy the closest existing template, change the measurements and register it in
`templates/__init__.py`.
