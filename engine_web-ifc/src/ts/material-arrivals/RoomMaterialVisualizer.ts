// src/ts/material-arrivals/RoomMaterialVisualizer.ts
import * as THREE from "three";

type IfcApiLike = {
  GetLineIDsWithType(modelID: number, type: number): any;
  GetLine(modelID: number, id: number): any;
};

export class RoomMaterialVisualizer {
  private roomBBoxByName = new Map<string, THREE.Box3>();
  private materialMeshes: THREE.Mesh[] = [];

constructor(
  private scene: THREE.Scene,
  private ifcAPI: IfcApiLike,
  private modelID: number,
  private IFCSPACE: number
) {}

  /** Build cache: IfcSpace.Name -> bbox (world coords). Call after geometry is loaded into the scene. */
  indexRoomsByIfcSpaceName() {
    this.roomBBoxByName.clear();

    // Helps bbox calculations be correct
    this.scene.updateMatrixWorld(true);

    const ids = this.ifcAPI.GetLineIDsWithType(this.modelID, this.IFCSPACE);
    const spaces: any[] = [];

    for (let i = 0; i < ids.size(); i++) {
      const id = ids.get(i);
      const space = this.ifcAPI.GetLine(this.modelID, id);
      if (space) spaces.push(space);
    }

    // Map expressID -> meshes in scene
    const meshesById = new Map<number, THREE.Mesh[]>();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if ((m as any).isMesh && m.userData && typeof m.userData.expressID === "number") {
        const id = m.userData.expressID as number;
        const arr = meshesById.get(id);
        if (arr) arr.push(m);
        else meshesById.set(id, [m]);
      }
    });

    for (const s of spaces) {
      const name = s?.Name?.value?.trim?.();
      if (!name) continue;

      const id = s.expressID as number;
      const meshes = meshesById.get(id);

      const bbox = new THREE.Box3();

      if (meshes && meshes.length > 0) {
        for (const m of meshes) bbox.expandByObject(m);
      } else {
        // fallback bbox (aligned to floor near scene center)
        const sceneBBox = new THREE.Box3().setFromObject(this.scene);
        const center = sceneBBox.getCenter(new THREE.Vector3());
        const minY = sceneBBox.min.y;
        const spread = 4;
        const idx = this.roomBBoxByName.size;

        bbox.set(
          new THREE.Vector3(center.x + idx * spread - 2, minY, center.z - 2),
          new THREE.Vector3(center.x + idx * spread + 2, minY + 2, center.z + 2)
        );
      }

      this.roomBBoxByName.set(this.cleanKey(name), bbox);
    }

    console.log("✅ Indexed rooms:", this.roomBBoxByName.size);
  }

  clearMaterialBoxes() {
    for (const m of this.materialMeshes) {
      this.scene.remove(m);
      m.geometry.dispose();
      // @ts-ignore
      m.material?.dispose && m.material.dispose();
    }
    this.materialMeshes.length = 0;
  }

  setMaterialsInRoom(roomName: string, count: number) {
    const key = this.cleanKey(roomName);
    const bbox = this.roomBBoxByName.get(key);
    if (!bbox) {
      console.warn(`Room not found: "${roomName}"`);
      console.warn("Known rooms:", this.getIndexedRoomNames());
      return;
    }
    this.clearMaterialBoxes();
    this.sprinkleBoxesInside(bbox, count);
  }

  getIndexedRoomNames() {
    return Array.from(this.roomBBoxByName.keys());
  }

  private sprinkleBoxesInside(bbox: THREE.Box3, count: number) {
    const center = bbox.getCenter(new THREE.Vector3());
    const size = new THREE.Vector3();
    bbox.getSize(size);

    const pad = 0.2;
    const sx = Math.max(size.x - pad, 0.2);
    const sz = Math.max(size.z - pad, 0.2);
    const floorY = bbox.min.y + 0.75;

    for (let i = 0; i < count; i++) {
      const p = new THREE.Vector3(
        center.x + (Math.random() - 0.5) * sx * 0.8,
        floorY,
        center.z + (Math.random() - 0.5) * sz * 0.8
      );
      this.addBoxAt(p, 0.25);
    }
  }

  private addBoxAt(position: THREE.Vector3, size = 0.25) {
    const geom = new THREE.BoxGeometry(size, size, size);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xff1493,
      metalness: 0.1,
      roughness: 0.7,
      emissive: 0xff0080,
      emissiveIntensity: 0.4,
    });

    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.copy(position);
    mesh.renderOrder = 999;
    // @ts-ignore
    mesh.material.depthTest = false;

    this.scene.add(mesh);
    this.materialMeshes.push(mesh);
  }

  private cleanKey(k: string) {
    return k.replace(/^'+|'+$/g, "").trim();
  }
}
