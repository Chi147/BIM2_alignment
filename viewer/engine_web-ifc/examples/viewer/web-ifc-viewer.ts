import * as THREE from "three";
import { IFCSPACE } from "../../dist/web-ifc-api";
import { RoomMaterialVisualizer } from "../../src/ts/material-arrivals/RoomMaterialVisualizer"; ///change1, add this line


import { IfcApplication } from "./../../src/ifc-schema";
import {
  IfcAPI,
  LogLevel,
  ms,
  Schemas,
  IFCUNITASSIGNMENT,
  IFCAXIS2PLACEMENT3D,
  IFCLENGTHMEASURE,
  IFCCARTESIANPOINT,
  IFCAXIS2PLACEMENT2D,
  IFCCIRCLEPROFILEDEF,
  IFCDIRECTION,
  IFCREAL,
  IFCPOSITIVELENGTHMEASURE,
  IFCCOLUMN,
  IFCEXTRUDEDAREASOLID,
  IFCGLOBALLYUNIQUEID,
  IFCLABEL,
  IFCIDENTIFIER,
} from "../../dist/web-ifc-api";
import { IfcThree } from "./web-ifc-three";
import { Init3DView, InitBasicScene, ClearScene, scene } from "./web-ifc-scene";
import * as Monaco from "monaco-editor";
import * as ts_decl from "./ts_src";
import * as ts from "typescript";
import { exampleCode } from "./example";

let ifcAPI = new IfcAPI();
ifcAPI.SetWasmPath("./");
let ifcThree = new IfcThree(ifcAPI);

let timeout = undefined;

let modelIDGlobal = -1;

// // Room name -> bounding box (in world coords)
// const roomBBoxByName = new Map<string, THREE.Box3>();

// // Keep “material boxes” so we can clear them
// const materialMeshes: THREE.Mesh[] = [];   //change2, comment these const lines out and add the line below 
let arrivalsVis: RoomMaterialVisualizer | null = null;



function Edited(monacoEditor: Monaco.editor.IStandaloneCodeEditor) {
  let code = monacoEditor.getValue();

  window.localStorage.setItem("code", code);
  console.log("Saved code...");
}

if (typeof window != "undefined") {
  //@ts-ignore
  window.InitMonaco = (monaco: any) => {
    console.log(ts_decl.ifc_schema);
    // validation settings
    monaco.languages.typescript.typescriptDefaults.setDiagnosticsOptions({
      noSemanticValidation: true,
      noSyntaxValidation: true,
    });

    // compiler options
    monaco.languages.typescript.typescriptDefaults.setCompilerOptions({
      target: monaco.languages.typescript.ScriptTarget.ES6,
      allowNonTsExtensions: true,
    });
    //@ts-ignore
    console.log(
      monaco.languages.typescript.typescriptDefaults.addExtraLib(
        ts_decl.ifc_schema
      )
    );
    console.log(
      monaco.languages.typescript.typescriptDefaults.addExtraLib(
        ts_decl.wifcapi
      )
    );
  };
}

function initMonacoEditor(monacoEditor: Monaco.editor.IStandaloneCodeEditor) {
  let item = window.localStorage.getItem("code");
  if (item) {
    monacoEditor.setValue(item);
  } else {
    monacoEditor.setValue(exampleCode);
  }

  monacoEditor.onDidChangeModelContent((e) => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => Edited(monacoEditor), 1000);
  });

  setTimeout(() => {
    Edited(monacoEditor);
  }, 1000);
}

if (typeof window != "undefined") {
  //@ts-ignore
  window.InitWebIfcViewer = async (
    monacoEditor: Monaco.editor.IStandaloneCodeEditor
  ) => {
    await ifcAPI.Init();
    initMonacoEditor(monacoEditor);
    const fileInput = document.getElementById("finput");
    fileInput.addEventListener("change", fileInputChanged);
    const codereset = document.getElementById("rcode");
    codereset.addEventListener("click", resetCode);
    const coderun = document.getElementById("runcode");
    coderun.addEventListener("click", runCode);
    const clearmem = document.getElementById("cmem");
    clearmem.addEventListener("click", clearMem);
    const changeLogLevelSelect = document.getElementById("logLevel");
    changeLogLevelSelect.addEventListener("change", changeLogLevel);
    Init3DView();
  };
}

async function changeLogLevel() {
  let fileInput = <HTMLInputElement>document.getElementById("logLevel");
  ifcAPI.SetLogLevel(fileInput.value);
  console.log("Log Level Set to:" + fileInput.value);
}

async function runCode() {
  let model = ifcAPI.CreateModel({ schema: Schemas.IFC4 });

  scene.clear();
  InitBasicScene();

  let code = window.localStorage.getItem("code");
  let compiled = ts.transpileModule(code, {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  });

  // this is where we do evil stuff
  {
    console.log(` --- Starting EVAL!`);
    eval(
      "(function (ifcAPI,IFCAXIS2PLACEMENT3D,IFCLENGTHMEASURE,IFCCARTESIANPOINT,IFCAXIS2PLACEMENT2D,IFCCIRCLEPROFILEDEF,IFCDIRECTION,IFCREAL,IFCPOSITIVELENGTHMEASURE,IFCCOLUMN,IFCEXTRUDEDAREASOLID,IFCGLOBALLYUNIQUEID,IFCLABEL,IFCIDENTIFIER) {" +
        compiled.outputText +
        "})"
    )(
      ifcAPI,
      IFCAXIS2PLACEMENT3D,
      IFCLENGTHMEASURE,
      IFCCARTESIANPOINT,
      IFCAXIS2PLACEMENT2D,
      IFCCIRCLEPROFILEDEF,
      IFCDIRECTION,
      IFCREAL,
      IFCPOSITIVELENGTHMEASURE,
      IFCCOLUMN,
      IFCEXTRUDEDAREASOLID,
      IFCGLOBALLYUNIQUEID,
      IFCLABEL,
      IFCIDENTIFIER
    );
    console.log(` --- Ending EVAL!`);
  }

  let ifcData = ifcAPI.SaveModel(model);
  let ifcDataString = new TextDecoder("ascii").decode(ifcData);

  //ifcAPI.CloseModel(model);

  let m2 = ifcAPI.OpenModel(ifcData);
  ifcThree.LoadAllGeometry(scene, m2);
}

async function resetCode() {
  window.localStorage.setItem("code", exampleCode);
  location.reload();
}

async function clearMem() {
  ClearScene();
  ifcAPI.Dispose();
  await ifcAPI.Init();
}

async function fileInputChanged() {
  let fileInput = <HTMLInputElement>document.getElementById("finput");
  if (fileInput.files.length == 0) return console.log("No files selected!");
  const file = fileInput.files[0];
  const reader = getFileReader(fileInput);
  reader.readAsArrayBuffer(file);
}

function getFileReader(fileInput: HTMLInputElement) {
  var reader = new FileReader();
  reader.onload = () => {
    const data = getData(reader);
    LoadModel(data);
    fileInput.value = "";
  };
  return reader;
}

function getData(reader: FileReader) {
  const startRead = ms();
  //@ts-ignore
  const data = new Uint8Array(reader.result);
  const readTime = ms() - startRead;
  console.log(`Reading took ${readTime} ms`);
  return data;
}

async function LoadModel(data: Uint8Array) {
  const start = ms();
  //  const modelID = ifcAPI.OpenModel(data, { COORDINATE_TO_ORIGIN: true, CIRCLE_SEGMENTS: 24, TOLERANCE_PLANE_INTERSECTION: 1.0E-04, TOLERANCE_PLANE_DEVIATION: 1.0E-04, TOLERANCE_BACK_DEVIATION_DISTANCE: 1.0E-04, TOLERANCE_INSIDE_OUTSIDE_PERIMETER: 1.0E-10, TOLERANCE_SCALAR_EQUALITY: 1.0E-04, PLANE_REFIT_ITERATIONS: 1, BOOLEAN_UNION_THRESHOLD: 150});
  const modelID = ifcAPI.OpenModel(data, {
    COORDINATE_TO_ORIGIN: true,
    CIRCLE_SEGMENTS: 6,
    TOLERANCE_PLANE_INTERSECTION: 1.0e-4,
    TOLERANCE_PLANE_DEVIATION: 1.0e-4,
    TOLERANCE_BACK_DEVIATION_DISTANCE: 1.0e-4,
    TOLERANCE_INSIDE_OUTSIDE_PERIMETER: 1.0e-10,
    TOLERANCE_SCALAR_EQUALITY: 1.0e-4,
    PLANE_REFIT_ITERATIONS: 3,
    BOOLEAN_UNION_THRESHOLD: 100,
  });
  modelIDGlobal = modelID;
  window.modelIDGlobal = modelID;
  // #1092 -> TOLERANCE_PLANE_INTERSECTION: 5.0E-02
  // #1256 -> TOLERANCE_PLANE_INTERSECTION: 1.0E-02, TOLERANCE_PLANE_DEVIATION: 1.0E-01, TOLERANCE_BACK_DEVIATION_DISTANCE: 1.0E-01
  // #1023 -> TOLERANCE_PLANE_DEVIATION: 1.0E-01, TOLERANCE_BACK_DEVIATION_DISTANCE: 1.0E-01
  // #540 -> TOLERANCE_PLANE_INTERSECTION: 1.0E-03, TOLERANCE_SCALAR_EQUALITY: 1.0E-01, PLANE_REFIT_ITERATIONS: 3
  // #1225 -> TOLERANCE_PLANE_DEVIATION: 1.0E-02, TOLERANCE_BACK_DEVIATION_DISTANCE: 1.0E-02, TOLERANCE_INSIDE_OUTSIDE_PERIMETER: 1.0E-02
  // #1506 -> TOLERANCE_PLANE_INTERSECTION = 1.0E-02, TOLERANCE_PLANE_DEVIATION = 1.0E-02, TOLERANCE_BACK_DEVIATION_DISTANCE = 1.0E-02;
  // #604 -> BOOLEAN_UNION_THRESHOLD: 50
  // Ferroflex, samMateo -> CIRCLE_SEGMENTS: 6
  const time = ms() - start;
  console.log(`Opening model took ${time} ms`);
  ifcThree.LoadAllGeometry(scene, modelID);
  // ---- Build a cache of IfcSpace bounding boxes by their Name ----
// ---- Build a cache of IfcSpace bounding boxes by their Name ----
// ---- Build a cache of IfcSpace bounding boxes by their Name ----

try {
  arrivalsVis = new RoomMaterialVisualizer(scene, ifcAPI, modelID, IFCSPACE);
  arrivalsVis.indexRoomsByIfcSpaceName();
} catch (err) {
  console.warn("Could not index IfcSpace bounding boxes:", err);
}


//change 4 comment section below and add section above
// try {
//   const ids = ifcAPI.GetLineIDsWithType(modelID, IFCSPACE);
//   const spaces: any[] = [];
//   for (let i = 0; i < ids.size(); i++) {
//     const id = ids.get(i);
//     const space = ifcAPI.GetLine(modelID, id);
//     if (space) spaces.push(space);
//   }

//   // Map expressID → meshes in the scene
//   const meshesById = new Map<number, THREE.Mesh[]>();
//   scene.traverse((o) => {
//     const m = o as THREE.Mesh;
//     if ((m as any).isMesh && m.userData && typeof m.userData.expressID === "number") {
//       const id = m.userData.expressID as number;
//       const arr = meshesById.get(id);
//       if (arr) arr.push(m);
//       else meshesById.set(id, [m]);
//     }
//   });

//   for (const s of spaces) {
//     const name = s?.Name?.value?.trim?.();
//     if (!name) continue;
//     const id = s.expressID as number;
//     const meshes = meshesById.get(id);

//     const bbox = new THREE.Box3();

//     if (meshes && meshes.length > 0) {
//       for (const m of meshes) bbox.expandByObject(m);
//     } else {
// // ✅ If there’s no geometry, place dummy bbox near building center
// // ✅ If there’s no geometry, create a dummy bbox aligned to the floor
// const sceneBBox = new THREE.Box3().setFromObject(scene);
// const center = sceneBBox.getCenter(new THREE.Vector3());
// const minY = sceneBBox.min.y; // floor height
// const spread = 4; // offset per room
// const idx = roomBBoxByName.size;

// bbox.set(
//   new THREE.Vector3(center.x + idx * spread - 2, minY, center.z - 2),
//   new THREE.Vector3(center.x + idx * spread + 2, minY + 2, center.z + 2)
// );

// console.warn(`No mesh for ${name}, using floor-aligned dummy bbox`, bbox);


//     }

//     roomBBoxByName.set(name, bbox);
//   }

//   // ✅ Normalize keys to remove quotes and trim whitespace
// for (const key of Array.from(roomBBoxByName.keys())) {
//   const clean = key.replace(/^'+|'+$/g, "").trim(); // remove stray quotes
//   if (clean !== key) {
//     const val = roomBBoxByName.get(key)!;
//     roomBBoxByName.delete(key);
//     roomBBoxByName.set(clean, val);
//   }
// }
// console.log("Cleaned roomBBoxByName keys:", Array.from(roomBBoxByName.keys()));


//   console.log("✅ Cached IfcSpace bboxes:", roomBBoxByName.size);
// } catch (err) {
//   console.warn("Could not cache IfcSpace bounding boxes:", err);
// }
// ----------------------------------------------------------------


  if (
    ifcAPI.GetModelSchema(modelID) == "IFC2X3" ||
    ifcAPI.GetModelSchema(modelID) == "IFC4" ||
    ifcAPI.GetModelSchema(modelID) == "IFC4X3_RC4"
  ) {
    //Example to get all types used in the model
    let types = await ifcAPI.GetAllTypesOfModel(modelID);
    if (types) {
      for (let i = 0; i < types.length; i++) {
        let type = types[i];
        //console.log(type);
        //console.log(type.typeID);
        //console.log(type.typeName);
      }
    }
  }

  try {
    // This function should activate only if we are in IFC4X3
    let alignments = await ifcAPI.GetAllAlignments(modelID);
    console.log("Alignments: ", alignments);
  } catch (error) {
    // Code to handle the error
    console.error("An error occurred:", error);
  }

  let lines = ifcAPI.GetLineIDsWithType(modelID, IFCUNITASSIGNMENT);
  //console.log(lines.size());
  for (let l = 0; l < lines.size(); l++) {
    //console.log(lines.get(l));
    let unitList = ifcAPI.GetLine(modelID, lines.get(l));
    //console.log(unitList);
    //console.log(unitList.Units);
    //console.log(unitList.Units.length);
    for (let u = 0; u < unitList.Units.length; u++) {
      //console.log(ifcAPI.GetLine(modelID, unitList.Units[u].value));
    }
  }
  //ifcAPI.CloseModel(modelID);
}

function clearMaterialBoxes() {
  for (const m of materialMeshes) {
    scene.remove(m);
    m.geometry.dispose();
    // @ts-ignore
    m.material?.dispose && m.material.dispose();
  }
  materialMeshes.length = 0;
}

function addBoxAt(position: THREE.Vector3, size = 0.25) {
  const geom = new THREE.BoxGeometry(size, size, size);
  const mat = new THREE.MeshStandardMaterial({
  color: 0xff1493, // 🎀 hot pink
  metalness: 0.1,
  roughness: 0.7,
  emissive: 0xff0080, // makes it glow slightly
  emissiveIntensity: 0.4,
});

  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.copy(position);
  mesh.renderOrder = 999;
  mesh.material.depthTest = false; // ensures they draw on top of glass
  scene.add(mesh);
  materialMeshes.push(mesh);
  return mesh;
}

function sprinkleBoxesInside(bbox: THREE.Box3, count: number) {
  const center = bbox.getCenter(new THREE.Vector3());
  const size = new THREE.Vector3();
  bbox.getSize(size);

  const pad = 0.2;
  const sx = Math.max(size.x - pad, 0.2);
  const sz = Math.max(size.z - pad, 0.2);
  const floorY = bbox.min.y + 0.75; // small offset above floor

  for (let i = 0; i < count; i++) {
    const p = new THREE.Vector3(
      center.x + (Math.random() - 0.5) * sx * 0.8,
      floorY, // ✅ fix Y to the floor
      center.z + (Math.random() - 0.5) * sz * 0.8
    );
    addBoxAt(p, 0.25);
  }
}



/** Highlight by slightly tinting meshes whose expressID is in ids */
function tintMeshesByIDs(ids: Set<number>, color = 0x00aaff, opacity = 0.25) {
  scene.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    // most meshes created by IfcThree have userData.expressID
    if ((mesh as any).isMesh && mesh.userData && ids.has(mesh.userData.expressID)) {
      const oldMat = mesh.material as THREE.Material | THREE.Material[];
      if (Array.isArray(oldMat)) return;
      const clone = oldMat.clone() as THREE.MeshStandardMaterial;
      clone.transparent = true;
      clone.opacity = opacity;
      clone.color = new THREE.Color(color);
      mesh.material = clone;
    }
  });
}

/** Public: call this when “materials arrive”. */
async function setMaterialsInRoom(roomName: string, count: number) {
  const key = roomName.trim();
  const bbox = roomBBoxByName.get(key);
  if (!bbox) {
    console.warn(`Room not found or not cached: ${roomName}`);
    return;
  }
  clearMaterialBoxes();
  sprinkleBoxesInside(bbox, count);
}

// Expose functions to the browser so index.html can call them
// if (typeof window !== "undefined") {
//   // @ts-ignore
//   window.__setMaterialsInRoom = (roomName: string, count: number) =>
//     setMaterialsInRoom(roomName, count);

//   // @ts-ignore
//   window.__clearMaterialBoxes = () => clearMaterialBoxes();
// }  //change4, comment these and replace by below section 
if (typeof window !== "undefined") {
  // @ts-ignore
  window.__setMaterialsInRoom = (roomName: string, count: number) =>
    arrivalsVis?.setMaterialsInRoom(roomName, count);

  // @ts-ignore
  window.__clearMaterialBoxes = () => arrivalsVis?.clearMaterialBoxes();

  // optional helper for debugging
  // @ts-ignore
  window.__listRooms = () => arrivalsVis?.getIndexedRoomNames();
}


// make viewer internals available in browser console
if (typeof window !== "undefined") {
  // @ts-ignore
  window.ifcAPI = ifcAPI;
  // @ts-ignore
  window.modelIDGlobal = modelIDGlobal;
  // @ts-ignore
  window.IFCSPACE = IFCSPACE;
  // @ts-ignore
  window.THREE = THREE;
  // @ts-ignore
  window.scene = scene;
}
