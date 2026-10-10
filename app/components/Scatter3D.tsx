"use client";
// 3D scatter plot via three.js. Static imports — three is an npm dependency,
// so the bundler includes it in the client chunk and the OrbitControls addon
// is just another static import.
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { COLORS, type ProjectedItem } from "../lib/state";

type ThreeState = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  pointsGroup: THREE.Group;
  raf: number;
  cleanup: () => void;
};

export function Scatter3D({
  items,
  newPoints,
}: {
  items: ProjectedItem[];
  newPoints: ProjectedItem[];
}) {
  const container = useRef<HTMLDivElement>(null);
  const state = useRef<ThreeState | null>(null);

  // One-time three.js setup; runs only in the browser.
  useEffect(() => {
    if (!container.current) return;
    const c = container.current;
    const w = c.clientWidth || 800;
    const h = c.clientHeight || 500;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 1000);
    camera.position.set(2.4, 2.0, 2.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(w, h);
    c.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AxesHelper(1.2));
    scene.add(new THREE.GridHelper(2.4, 12, 0x2a3142, 0x1d2230));

    const hint = document.createElement("div");
    hint.className = "three-hint";
    hint.textContent = "drag to rotate · scroll to zoom · right-drag to pan";
    c.appendChild(hint);

    const pointsGroup = new THREE.Group();
    scene.add(pointsGroup);

    const onResize = () => {
      if (!container.current) return;
      const nw = container.current.clientWidth;
      const nh = container.current.clientHeight;
      if (!nw || !nh) return;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener("resize", onResize);

    let raf = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    state.current = {
      scene,
      camera,
      renderer,
      controls,
      pointsGroup,
      raf,
      cleanup: () => {
        cancelAnimationFrame(raf);
        window.removeEventListener("resize", onResize);
        renderer.dispose();
        if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
        if (hint.parentNode) hint.parentNode.removeChild(hint);
      },
    };

    return () => {
      state.current?.cleanup();
      state.current = null;
    };
  }, []);

  // Re-render the points whenever data changes.
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    const { pointsGroup, camera, controls } = s;

    // Dispose old points
    while (pointsGroup.children.length) {
      const c = pointsGroup.children.pop();
      if (c instanceof THREE.Points) {
        c.geometry.dispose();
        const m = c.material;
        if (Array.isArray(m)) m.forEach((mm) => mm.dispose());
        else m.dispose();
      }
    }

    const allPts: { x: number; y: number; z: number }[] = [];
    for (const it of items) allPts.push({ x: it.x, y: it.y, z: it.z });
    for (const p of newPoints) allPts.push({ x: p.x, y: p.y, z: p.z });
    if (!allPts.length) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of allPts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
      if (p.z < minZ) minZ = p.z;
      if (p.z > maxZ) maxZ = p.z;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const range = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
    const norm = (v: number) => v / range;

    if (items.length) {
      const geom = new THREE.BufferGeometry();
      const pos = new Float32Array(items.length * 3);
      for (let i = 0; i < items.length; i++) {
        pos[i * 3]     = norm(items[i].x - cx);
        pos[i * 3 + 1] = norm(items[i].y - cy);
        pos[i * 3 + 2] = norm(items[i].z - cz);
      }
      geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      pointsGroup.add(new THREE.Points(geom, new THREE.PointsMaterial({
        color: COLORS.dataset, size: 0.05, sizeAttenuation: true,
      })));
    }
    if (newPoints.length) {
      const geom = new THREE.BufferGeometry();
      const pos = new Float32Array(newPoints.length * 3);
      for (let i = 0; i < newPoints.length; i++) {
        pos[i * 3]     = norm(newPoints[i].x - cx);
        pos[i * 3 + 1] = norm(newPoints[i].y - cy);
        pos[i * 3 + 2] = norm(newPoints[i].z - cz);
      }
      geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      pointsGroup.add(new THREE.Points(geom, new THREE.PointsMaterial({
        color: COLORS.newPoint, size: 0.08, sizeAttenuation: true,
      })));
    }

    const target = new THREE.Vector3(0, 0, 0);
    const desiredDist = range * 2.4;
    camera.position.set(desiredDist, desiredDist * 0.8, desiredDist);
    camera.lookAt(target);
    controls.target.copy(target);
    controls.update();
  }, [items, newPoints]);

  return <div ref={container} className="three-container" style={{ position: "absolute", inset: 12 }} />;
}