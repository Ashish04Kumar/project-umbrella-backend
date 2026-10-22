/*
 * Robot plugin: the Umbrella 3D robot head for the web playground.
 *
 * A port of the Synq app's Buddy tab robot (mobile/src/features/buddy/robot/: robotModel.ts,
 * robotState.ts, RobotFace.tsx, RobotCanvas.tsx) from react-three-fiber to plain three.js:
 * the same model and region colours, the same face shaders (eyes with curved lids, scan-line
 * screen, blush, mouth), the same saccade gaze, blinks and head motion. On top of the app's
 * five body states (idle, listening, thinking, speaking, error) it draws the reply moods the
 * LLM picks: happy, curious, concerned, playful (winks), surprised, excited and sad.
 *
 * Plugin contract (see playground.html): mount(container, bus) and the bus sends
 *   "state"      idle | listening | thinking | speaking | error
 *   "expression" a reply mood (shown for a while after the reply, longer while speaking)
 *   "level"      0..1 loudness: the user's mic while listening, Umbrella's voice while speaking
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const MODEL_URL = new URL("./gg2-head.glb", import.meta.url).href;

// --- Model geometry and colours (from robotModel.ts) ----------------------------------------------

const FACE = { center: { x: 0.15, y: 0.425, z: 0.2645 }, halfWidth: 0.42, halfHeight: 0.385, radius: 0.24 };
const MODEL_CENTER = { x: 0.153, y: 0.45, z: 0 };

const SHELL = new THREE.Color("#F6CDB6");
const SHELL_SHADE = new THREE.Color("#E8AE97");
const SCREEN = new THREE.Color("#2B0A12");
const NECK = new THREE.Color("#5A2630");
const EAR = new THREE.Color("#DE9E90");
const ANTENNA_TIP = new THREE.Color("#FF8A7A");

function insideRoundedRect(x, y, cx, cy, halfW, halfH, radius) {
  const dx = Math.max(Math.abs(x - cx) - (halfW - radius), 0);
  const dy = Math.max(Math.abs(y - cy) - (halfH - radius), 0);
  return Math.abs(x - cx) <= halfW && Math.abs(y - cy) <= halfH && dx * dx + dy * dy <= radius * radius;
}

function regionColor(x, y, z, normalZ) {
  const facesFront = z > 0.2 && normalZ > 0.5;
  // The screen is painted maroon a little inside the face panel, so colour blending stays hidden under it.
  const inset = 0.03;
  if (facesFront && insideRoundedRect(x, y, FACE.center.x, FACE.center.y, FACE.halfWidth - inset, FACE.halfHeight - inset, FACE.radius - inset)) return SCREEN;
  const isEar = (x < -0.4 || x > 0.7) && y > 0.12;
  if (isEar) return y > 0.78 ? ANTENNA_TIP : EAR;
  if (y > -0.13 && y < -0.03 && Math.abs(x - FACE.center.x) < 0.2) return NECK;
  if (x < -0.62 || x > 0.66 || y < -0.7) return SHELL_SHADE;
  return SHELL;
}

function prepareRobotGeometry(geometry) {
  geometry.computeVertexNormals();
  const position = geometry.getAttribute("position");
  const normal = geometry.getAttribute("normal");
  const colors = new Float32Array(position.count * 3);
  for (let i = 0; i < position.count; i += 1) {
    const color = regionColor(position.getX(i), position.getY(i), position.getZ(i), normal.getZ(i));
    colors[i * 3] = color.r; colors[i * 3 + 1] = color.g; colors[i * 3 + 2] = color.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

// --- States, looks and expressions (from robotState.ts, plus the reply moods) ----------------------

const LOOKS = {
  idle: { label: "Here with you", tint: "#FF8A7A", glow: 0.45, motion: 1 },
  listening: { label: "Listening", tint: "#FFB199", glow: 0.65, motion: 0.8 },
  thinking: { label: "Thinking", tint: "#C9A7FF", glow: 0.6, motion: 0.6 },
  speaking: { label: "Speaking", tint: "#FF7A8A", glow: 0.85, motion: 1.1 },
  error: { label: "Something went wrong", tint: "#FF4D5E", glow: 0.4, motion: 0.5 },
};

// Lid/mouth/head values per state (same meaning as in the app: see robotState.ts).
// mouthOpen: how open the mouth rests (surprise "o"); bounce: how lively the float is.
const base = { mouthOpen: 0, bounce: 1 };
const EXPRESSIONS = {
  idle: { ...base, upperLid: 0.02, lowerLid: 0, lidTilt: 0, squintSide: 0, pupil: 1, mouthCurve: 1, mouthWidth: 1, mouthShift: 0, headRoll: 0, headPitch: 0, blink: [2.6, 5.5], gaze: [1.2, 3.4] },
  listening: { ...base, upperLid: 0, lowerLid: 0.04, lidTilt: 0.04, squintSide: 0, pupil: 1.12, mouthCurve: 0.75, mouthWidth: 0.85, mouthShift: 0, headRoll: 0.03, headPitch: 0.02, blink: [3, 6], gaze: [0.5, 1.3] },
  thinking: { ...base, upperLid: 0.26, lowerLid: 0.12, lidTilt: -0.08, squintSide: 0.12, pupil: 0.95, mouthCurve: 0.1, mouthWidth: 0.62, mouthShift: 0.03, headRoll: 0.07, headPitch: -0.05, blink: [4, 7], gaze: [0.9, 2.1] },
  speaking: { ...base, upperLid: 0.04, lowerLid: 0.22, lidTilt: 0, squintSide: 0, pupil: 1.05, mouthCurve: 1, mouthWidth: 1, mouthShift: 0, headRoll: 0, headPitch: 0, blink: [2.8, 4.8], gaze: [0.5, 1.4] },
  error: { ...base, upperLid: 0.2, lowerLid: 0.04, lidTilt: 0.32, squintSide: 0, pupil: 0.85, mouthCurve: -0.7, mouthWidth: 0.7, mouthShift: 0, headRoll: -0.04, headPitch: 0.05, blink: [2, 3.6], gaze: [1.4, 2.6] },
};

// Reply moods (new for the web robot). Shown after a reply, and while speaking (the mouth still follows the voice).
const MOODS = {
  happy: { ...base, upperLid: 0.02, lowerLid: 0.32, lidTilt: 0, squintSide: 0, pupil: 1.05, mouthCurve: 1, mouthWidth: 1.12, mouthShift: 0, headRoll: 0.03, headPitch: 0, blink: [2.6, 5], gaze: [1, 2.6] },
  curious: { ...base, upperLid: 0, lowerLid: 0.02, lidTilt: 0.1, squintSide: 0.08, pupil: 1.2, mouthCurve: 0.35, mouthWidth: 0.7, mouthShift: 0.025, headRoll: 0.13, headPitch: -0.02, blink: [3, 6], gaze: [0.6, 1.6] },
  concerned: { ...base, upperLid: 0.14, lowerLid: 0.06, lidTilt: 0.3, squintSide: 0, pupil: 0.95, mouthCurve: -0.35, mouthWidth: 0.7, mouthShift: 0, headRoll: -0.05, headPitch: 0.05, blink: [2.6, 4.6], gaze: [1, 2.4] },
  playful: { ...base, upperLid: 0.04, lowerLid: 0.24, lidTilt: 0, squintSide: 0, pupil: 1.05, mouthCurve: 1, mouthWidth: 1, mouthShift: 0.03, headRoll: -0.09, headPitch: 0, blink: [3, 6], gaze: [0.8, 2], bounce: 1.3 },
  surprised: { ...base, upperLid: 0, lowerLid: 0, lidTilt: 0.05, squintSide: 0, pupil: 0.78, mouthCurve: 0, mouthWidth: 0.55, mouthShift: 0, headRoll: 0, headPitch: -0.06, blink: [3.5, 6], gaze: [1.2, 2.5], mouthOpen: 0.45 },
  excited: { ...base, upperLid: 0, lowerLid: 0.36, lidTilt: 0, squintSide: 0, pupil: 1.15, mouthCurve: 1, mouthWidth: 1.25, mouthShift: 0, headRoll: 0.04, headPitch: -0.03, blink: [2.2, 4], gaze: [0.5, 1.2], mouthOpen: 0.25, bounce: 1.8 },
  sad: { ...base, upperLid: 0.3, lowerLid: 0.04, lidTilt: 0.42, squintSide: 0, pupil: 0.92, mouthCurve: -0.8, mouthWidth: 0.62, mouthShift: 0, headRoll: -0.06, headPitch: 0.1, blink: [3, 5.5], gaze: [1.6, 3], bounce: 0.5 },
};
const MOOD_TINT = { happy: "#FFB27A", curious: "#8FD3FF", concerned: "#C9A7FF", playful: "#FF9AD5", surprised: "#FFE27A", excited: "#FF7AB8", sad: "#8FA8FF" };
const MOOD_LABEL = { happy: "Happy", curious: "Curious", concerned: "Concerned", playful: "Playful", surprised: "Surprised", excited: "Excited", sad: "Feeling it with you" };
const MOOD_SECONDS = 8;   // how long a mood stays on the face after the reply

function nextGaze(look, secondsInState, step, random) {
  const jitter = (size) => (random() - 0.5) * 2 * size;
  switch (look) {
    case "listening": case "curious": case "surprised":
      return { x: jitter(0.007), y: jitter(0.005) };   // fixed on you, tiny micro-movements
    case "thinking":
      if (random() < 0.22) return { x: 0.032 * (step % 2 === 0 ? 1 : -1), y: 0.004 };
      return step % 2 === 0 ? { x: 0.03 + jitter(0.004), y: 0.03 + jitter(0.004) } : { x: -0.026 + jitter(0.004), y: 0.033 + jitter(0.004) };
    case "speaking": case "happy": case "excited": case "playful":
      return random() < 0.25 ? { x: 0.026 * (random() < 0.5 ? 1 : -1), y: jitter(0.01) } : { x: jitter(0.012), y: jitter(0.008) };
    case "sad": case "concerned":
      return random() < 0.3 ? { x: jitter(0.01), y: 0.0 } : { x: jitter(0.012), y: -0.02 };   // mostly down
    case "error":
      if (secondsInState < 1.4) return [{ x: -0.032, y: 0 }, { x: 0.032, y: 0 }, { x: 0, y: -0.022 }][Math.min(step, 2)];
      return random() < 0.3 ? { x: jitter(0.01), y: 0.01 } : { x: jitter(0.012), y: -0.02 };
    default:
      return random() < 0.3 ? { x: 0, y: 0 } : { x: jitter(0.03), y: jitter(0.02) };
  }
}

const smoothToward = (current, target, dt, speed) => current + (target - current) * (1 - Math.exp(-dt * speed));
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const between = ([min, max]) => min + Math.random() * (max - min);
const blinkAmount = (p) => (p < 0 || p >= 1 ? 0 : p < 0.4 ? p / 0.4 : 1 - (p - 0.4) / 0.6);

// --- Shaders (from RobotFace.tsx and RobotCanvas.tsx) ----------------------------------------------

const POS_VERTEX = `
  varying vec2 vPos;
  void main() { vPos = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;
const UV_VERTEX = `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;
const SCREEN_FRAGMENT = `
  uniform float uTime; uniform float uGlow; uniform float uGlitch; uniform vec3 uTint; uniform vec2 uHalf;
  varying vec2 vPos;
  void main() {
    vec2 p = vPos / uHalf;
    float centre = 1.0 - clamp(length(p * vec2(0.9, 1.0)), 0.0, 1.0);
    vec3 base = mix(vec3(0.075, 0.012, 0.03), vec3(0.2, 0.04, 0.075), centre);
    float lines = 0.5 + 0.5 * sin(vPos.y * 260.0 + uTime * 4.0);
    base *= 0.9 + 0.1 * lines;
    float sweepY = 1.2 - mod(uTime * 0.35, 2.6);
    float sweep = exp(-pow((p.y - sweepY) * 6.0, 2.0)) * 0.12;
    float edge = smoothstep(0.55, 1.0, max(abs(p.x), abs(p.y)));
    float band = step(0.55, fract(vPos.y * 14.0 + uTime * 9.0));
    vec3 color = base + uTint * (sweep + edge * 0.35 * uGlow + centre * 0.12 * uGlow + band * uGlitch * 0.22);
    gl_FragColor = vec4(color, 1.0);
  }
`;
const EYE_FRAGMENT = `
  uniform float uR; uniform float uH; uniform float uUpper; uniform float uLower; uniform float uTilt;
  uniform float uSide; uniform vec2 uGaze; uniform float uPupil;
  varying vec2 vPos;
  const float AA = 0.003;
  void main() {
    float x = vPos.x; float y = vPos.y; float nx = x / uR;
    float lowerCentre = -uH + 2.0 * uH * uLower;
    float yLower = -uH + 2.0 * uH * uLower * (1.0 - 0.35 * nx * nx);
    float u = clamp(uUpper, 0.0, 1.0);
    float apex = mix(uH * 1.02, lowerCentre, u);
    float bend = uH * 0.35 * (1.0 - 1.6 * u * u);
    float yUpper = apex - bend * nx * nx + uTilt * (-uSide) * x;
    float shape = length(vec2(nx, y / uH));
    float inEye = 1.0 - smoothstep(1.0 - AA / uR, 1.0 + AA / uR, shape);
    float belowUpper = 1.0 - smoothstep(yUpper - AA, yUpper + AA, y);
    float aboveLower = smoothstep(yLower - AA, yLower + AA, y);
    float open = inEye * belowUpper * aboveLower * (1.0 - smoothstep(0.86, 0.99, u));
    vec3 color = vec3(1.0, 0.969, 0.953);
    float underLid = max(yUpper - y, 0.0);
    color *= 1.0 - 0.16 * exp(-underLid / (0.16 * uH));
    float pupilR = 0.038 * uPupil;
    float pupil = 1.0 - smoothstep(pupilR - AA, pupilR + AA, length(vPos - uGaze));
    color = mix(color, vec3(0.227, 0.051, 0.094), pupil);
    float shineR = 0.011 * uPupil;
    float shine = 1.0 - smoothstep(shineR - AA, shineR + AA, length(vPos - uGaze - vec2(0.014, 0.016) * uPupil));
    color = mix(color, vec3(1.0), shine);
    float lineFade = smoothstep(0.72, 0.96, uUpper);
    float onLine = 1.0 - smoothstep(0.008 - AA, 0.008 + AA, abs(y - yUpper));
    float lineWidth = 1.0 - smoothstep(uR * 0.9 - AA, uR * 0.9 + AA, abs(x));
    float line = lineFade * onLine * lineWidth;
    float alpha = max(open, line);
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(open >= line ? color : vec3(1.0, 0.93, 0.9), alpha);
  }
`;
const EYE_HALO_FRAGMENT = `
  uniform float uStrength; uniform vec3 uTint; varying vec2 vUv;
  void main() {
    float d = length(vUv - 0.5) * 2.0;
    float fade = 1.0 - smoothstep(0.62, 1.0, d);
    gl_FragColor = vec4(mix(vec3(1.0, 0.86, 0.82), uTint, 0.5), fade * uStrength);
  }
`;
const BACK_HALO_FRAGMENT = `
  uniform float uStrength; uniform vec3 uTint; varying vec2 vUv;
  void main() {
    float d = length(vUv - 0.5) * 2.0;
    float glow = pow(clamp(1.0 - d, 0.0, 1.0), 2.2) * uStrength;
    gl_FragColor = vec4(mix(vec3(0.97, 0.66, 0.66), uTint, 0.35), glow);
  }
`;
const MAX_HALO = 0.32;

function roundedRect(halfW, halfH, radius) {
  const shape = new THREE.Shape();
  shape.moveTo(-halfW + radius, -halfH);
  shape.lineTo(halfW - radius, -halfH);
  shape.absarc(halfW - radius, -halfH + radius, radius, -Math.PI / 2, 0, false);
  shape.lineTo(halfW, halfH - radius);
  shape.absarc(halfW - radius, halfH - radius, radius, 0, Math.PI / 2, false);
  shape.lineTo(-halfW + radius, halfH);
  shape.absarc(-halfW + radius, halfH - radius, radius, Math.PI / 2, Math.PI, false);
  shape.lineTo(-halfW, -halfH + radius);
  shape.absarc(-halfW + radius, -halfH + radius, radius, Math.PI, Math.PI * 1.5, false);
  return shape;
}

// Face layout (face units, centre of the screen = 0,0).
const EYE_X = 0.15, EYE_Y = 0.07, EYE_R = 0.078, EYE_H = EYE_R * 1.15, MOUTH_Y = -0.15, TOP_Z = 0.016;

function buildFace() {
  const face = new THREE.Group();
  face.position.set(FACE.center.x, FACE.center.y, FACE.center.z);

  const screenMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 }, uGlow: { value: 0.5 }, uGlitch: { value: 0 },
      uTint: { value: new THREE.Color(LOOKS.idle.tint) }, uHalf: { value: new THREE.Vector2(FACE.halfWidth, FACE.halfHeight) },
    },
    vertexShader: POS_VERTEX, fragmentShader: SCREEN_FRAGMENT,
  });
  face.add(new THREE.Mesh(new THREE.ShapeGeometry(roundedRect(FACE.halfWidth, FACE.halfHeight, FACE.radius), 16), screenMaterial));

  const eyes = [], eyeGlow = [];
  [-EYE_X, EYE_X].forEach((x, index) => {
    const group = new THREE.Group();
    group.position.set(x, EYE_Y, 0);
    const eye = new THREE.ShaderMaterial({
      uniforms: {
        uR: { value: EYE_R }, uH: { value: EYE_H }, uUpper: { value: 0.02 }, uLower: { value: 0 }, uTilt: { value: 0 },
        uSide: { value: index === 0 ? -1 : 1 }, uGaze: { value: new THREE.Vector2() }, uPupil: { value: 1 },
      },
      vertexShader: POS_VERTEX, fragmentShader: EYE_FRAGMENT, transparent: true, depthWrite: false,
    });
    const eyeMesh = new THREE.Mesh(new THREE.PlaneGeometry(EYE_R * 2 + 0.03, EYE_H * 2 + 0.03), eye);
    eyeMesh.position.z = 0.006;
    const halo = new THREE.ShaderMaterial({
      uniforms: { uStrength: { value: 0.2 }, uTint: { value: new THREE.Color(LOOKS.idle.tint) } },
      vertexShader: UV_VERTEX, fragmentShader: EYE_HALO_FRAGMENT, transparent: true, depthWrite: false,
    });
    const haloMesh = new THREE.Mesh(new THREE.CircleGeometry(0.115, 32), halo);
    haloMesh.position.z = 0.014; haloMesh.scale.set(1, 1.15, 1);
    group.add(eyeMesh, haloMesh);
    face.add(group);
    eyes.push(eye); eyeGlow.push(halo);
  });

  const blush = [-0.255, 0.255].map((x) => {
    const material = new THREE.MeshBasicMaterial({ color: "#FF8FA3", transparent: true, opacity: 0.45, depthWrite: false });
    const mesh = new THREE.Mesh(new THREE.CircleGeometry(0.042, 24), material);
    mesh.position.set(x, -0.095, TOP_Z); mesh.scale.set(1.3, 0.75, 1);
    face.add(mesh);
    return material;
  });

  const mouthGroup = new THREE.Group();
  mouthGroup.position.set(0, MOUTH_Y, TOP_Z);
  const mouthArcMaterial = new THREE.MeshBasicMaterial({ color: "#FFE3DA", transparent: true });
  const mouthArc = new THREE.Mesh(new THREE.TorusGeometry(0.062, 0.011, 8, 28, Math.PI), mouthArcMaterial);
  mouthArc.rotation.z = Math.PI;   // half a torus upside down = the smile line
  const mouthLineMaterial = new THREE.MeshBasicMaterial({ color: "#FFE3DA", transparent: true, opacity: 0 });
  const mouthLine = new THREE.Mesh(new THREE.CapsuleGeometry(0.009, 0.1, 4, 12), mouthLineMaterial);
  mouthLine.position.y = -0.03; mouthLine.rotation.z = Math.PI / 2; mouthLine.visible = false;
  const mouthOpen = new THREE.Group();
  mouthOpen.position.y = -0.03; mouthOpen.visible = false;
  const lips = new THREE.Mesh(new THREE.CircleGeometry(0.05, 32), new THREE.MeshBasicMaterial({ color: "#FF8A7A" }));
  const inside = new THREE.Mesh(new THREE.CircleGeometry(0.05, 32), new THREE.MeshBasicMaterial({ color: "#4A0E1C" }));
  inside.position.set(0, -0.004, 0.001); inside.scale.set(0.78, 0.7, 1);
  mouthOpen.add(lips, inside);
  mouthGroup.add(mouthArc, mouthLine, mouthOpen);
  face.add(mouthGroup);

  return { face, screenMaterial, eyes, eyeGlow, blush, mouthGroup, mouthArc, mouthArcMaterial, mouthLine, mouthLineMaterial, mouthOpen };
}

// --- The plugin ------------------------------------------------------------------------------------

export function mount(container, bus) {
  container.classList.add("robot-plugin");
  const label = document.createElement("div");
  label.className = "robot-label";
  label.textContent = "Loading Umbrella…";
  container.append(label);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (error) {
    label.textContent = "The 3D robot needs WebGL, which is off in this browser.";
    return { unmount() { container.textContent = ""; } };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.NoToneMapping;   // like the app's `flat` canvas: brand colours stay exact
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  container.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  camera.position.set(0, 0.05, 3.2);   // a little closer than the app (3.9): the stage is a short strip
  scene.add(new THREE.HemisphereLight("#FFF4EE", "#7A3A44", 0.9));
  [[[2.2, 3, 4], 2.1, "#FFF1EA"], [[-3, 1.5, -2.5], 1.4, "#FF9C8E"], [[-2.5, -1, 3], 0.5, "#FFD6CC"]].forEach(([pos, intensity, color]) => {
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.set(...pos);
    scene.add(light);
  });

  const backHalo = new THREE.ShaderMaterial({
    uniforms: { uStrength: { value: 0.35 }, uTint: { value: new THREE.Color(LOOKS.idle.tint) } },
    vertexShader: UV_VERTEX, fragmentShader: BACK_HALO_FRAGMENT, transparent: true, depthWrite: false,
  });
  const haloMesh = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 4.2), backHalo);
  haloMesh.position.z = -1.2;
  scene.add(haloMesh);

  const body = new THREE.Group();
  const offset = new THREE.Group();
  offset.position.set(-MODEL_CENTER.x, -MODEL_CENTER.y, -MODEL_CENTER.z);
  body.add(offset);
  scene.add(body);
  const parts = buildFace();
  offset.add(parts.face);

  new GLTFLoader().loadAsync(MODEL_URL).then((gltf) => {
    let geometry = null;
    gltf.scene.traverse((child) => { if (!geometry && child.isMesh) geometry = child.geometry; });
    if (!geometry) throw new Error("The robot model has no mesh.");
    offset.add(new THREE.Mesh(prepareRobotGeometry(geometry), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.36, metalness: 0.06 })));
    updateLabel();
  }).catch((error) => { label.textContent = "Robot could not load: " + error.message; signals.state = "error"; });

  // What the playground tells us (bus), and how long a mood stays.
  const signals = { state: "idle", level: 0, mood: null, moodUntil: 0 };
  const clock = new THREE.Clock();
  const now = () => clock.getElapsedTime();

  function activeMood() {
    if (!signals.mood) return null;
    if (signals.state === "speaking") signals.moodUntil = Math.max(signals.moodUntil, now() + 2);   // keep it while talking
    return now() < signals.moodUntil ? signals.mood : null;
  }
  // Listening, thinking and error always win; otherwise the reply's mood shows (if any).
  function currentLook() {
    if (signals.state === "listening" || signals.state === "thinking" || signals.state === "error") return signals.state;
    return activeMood() || signals.state;
  }
  function updateLabel() {
    const look = currentLook();
    label.textContent = MOOD_LABEL[look] ? `${LOOKS[signals.state].label} · ${MOOD_LABEL[look]}` : LOOKS[signals.state].label;
  }

  const off = [
    bus.on("state", (state) => { if (LOOKS[state]) { signals.state = state; if (state === "listening" || state === "thinking" || state === "error") signals.mood = null; updateLabel(); } }),
    bus.on("expression", (mood) => { signals.mood = MOODS[mood] ? mood : null; signals.moodUntil = now() + MOOD_SECONDS; updateLabel(); }),
    bus.on("level", (level) => { signals.level = clamp(Number(level) || 0, 0, 1); }),
  ];

  const anim = {
    look: "idle", lookSince: 0, glow: 0.5, upper: 0.02, lower: 0, tilt: 0, squint: 0, pupil: 1,
    gazeX: 0, gazeY: 0, targetX: 0, targetY: 0, gazeStep: 0, nextGaze: 1, blinkStart: -10, nextBlink: 2.5,
    mouthOpen: 0, mouthCurve: 1, mouthWidth: 1, mouthShift: 0, wasLoud: false, loudSince: 0, nextGlitch: 0, glitchUntil: 0,
    tint: new THREE.Color(LOOKS.idle.tint), targetTint: new THREE.Color(),
    speed: 1, phase: 0, roll: 0, headPitch: 0, followX: 0, followY: 0, nod: 0, halo: 0.35, bounce: 1, lastLabel: "",
  };
  let level = 0;   // smoothed

  function frame(dt, time) {
    const a = anim;
    const state = signals.state;
    const look = currentLook();
    const expression = EXPRESSIONS[look] || MOODS[look];
    const stateLook = LOOKS[state];
    level = smoothToward(level, signals.level, dt, 18);
    if (look !== a.look) {
      a.look = look; a.lookSince = time; a.gazeStep = 0; a.nextGaze = time;
      if (look === "speaking" || look === "listening" || MOODS[look]) a.nextBlink = time + 0.12;   // a "hello" blink
      if (look === "error") a.nextGlitch = time;
      updateLabel();
    }
    const inLook = time - a.lookSince;

    // Screen glow and colour.
    const boost = state === "listening" || state === "speaking" ? level * 0.3 : 0;
    a.glow = smoothToward(a.glow, clamp(stateLook.glow + boost + Math.sin(time * 1.6) * 0.05, 0, 1), dt, 3);
    a.targetTint.set(MOOD_TINT[look] || stateLook.tint);
    a.tint.lerp(a.targetTint, 1 - Math.exp(-dt * 4));
    let glitch = 0;
    if (look === "error") {
      if (time > a.nextGlitch) { a.glitchUntil = time + 0.16; a.nextGlitch = time + 2.6 + Math.random() * 1.8; }
      glitch = time < a.glitchUntil ? 1 : 0;
    }
    parts.face.position.x = FACE.center.x + (glitch ? Math.sin(time * 90) * 0.006 : 0);
    const su = parts.screenMaterial.uniforms;
    su.uTime.value = time; su.uGlow.value = a.glow; su.uGlitch.value = glitch; su.uTint.value.copy(a.tint);

    // Gaze: a quick jump to a new point, then hold.
    if (time > a.nextGaze) {
      const target = nextGaze(look, inLook, a.gazeStep, Math.random);
      a.targetX = clamp(target.x, -0.032, 0.032); a.targetY = clamp(target.y, -0.03, 0.035);
      a.gazeStep += 1;
      a.nextGaze = time + (look === "error" && inLook < 1.4 ? 0.42 : between(expression.gaze));
    }
    a.gazeX = smoothToward(a.gazeX, a.targetX, dt, 26);
    a.gazeY = smoothToward(a.gazeY, a.targetY, dt, 26);
    const bob = state === "speaking" ? level * 0.006 : 0;

    // Blinks, and a blink at the end of a spoken phrase.
    const loud = state === "speaking" && level > 0.12;
    if (loud && !a.wasLoud) a.loudSince = time;
    if (!loud && a.wasLoud && time - a.loudSince > 0.8 && Math.random() < 0.5) a.nextBlink = Math.min(a.nextBlink, time + 0.05);
    a.wasLoud = loud;
    if (time > a.nextBlink) { a.blinkStart = time; a.nextBlink = Math.random() < 0.18 ? time + 0.28 : time + between(expression.blink); }
    const blink = blinkAmount((time - a.blinkStart) / 0.15);

    // Lids.
    const engaged = state === "listening" ? level * 0.12 : state === "speaking" ? level * 0.1 : 0;
    // Playful: a wink of the right eye every few seconds.
    const wink = look === "playful" ? ((time - a.lookSince) % 3.2 < 0.55 ? 0.96 : 0) : 0;
    a.upper = smoothToward(a.upper, expression.upperLid, dt, 7);
    a.lower = smoothToward(a.lower, expression.lowerLid + engaged, dt, 7);
    a.tilt = smoothToward(a.tilt, expression.lidTilt, dt, 6);
    a.squint = smoothToward(a.squint, Math.max(expression.squintSide, wink), dt, wink ? 18 : 6);
    a.pupil = smoothToward(a.pupil, expression.pupil, dt, 5);
    parts.eyes.forEach((material, eye) => {
      const u = material.uniforms;
      const side = eye === 0 ? -1 : 1;
      u.uUpper.value = clamp(Math.max(a.upper + (side > 0 ? a.squint : 0), blink), 0, 1);
      u.uLower.value = clamp(a.lower + blink * 0.12, 0, 0.6);
      u.uTilt.value = a.tilt * 0.8;
      u.uGaze.value.set(a.gazeX, a.gazeY + bob);
      u.uPupil.value = a.pupil;
    });
    for (const material of parts.eyeGlow) {
      material.uniforms.uStrength.value = Math.min(MAX_HALO, 0.12 + a.glow * 0.2) * (1 - blink * 0.6);
      material.uniforms.uTint.value.copy(a.tint);
    }
    for (const material of parts.blush) material.opacity = look === "error" ? 0.14 : look === "sad" ? 0.22 : 0.36 + a.glow * 0.14;

    // Mouth: follows the voice while speaking; otherwise rests in the expression's shape.
    const wobble = Math.sin(time * 23) * Math.sin(time * 7.1) * 0.12;
    const voiceOpen = state === "speaking" && level > 0.02 ? clamp(level * 1.5 + wobble, 0, 1) : 0;
    const openTarget = Math.max(voiceOpen, expression.mouthOpen || 0);
    a.mouthOpen = smoothToward(a.mouthOpen, openTarget, dt, openTarget > a.mouthOpen ? 28 : 11);
    a.mouthCurve = smoothToward(a.mouthCurve, expression.mouthCurve, dt, 6);
    a.mouthWidth = smoothToward(a.mouthWidth, expression.mouthWidth, dt, 6);
    const hmm = look === "thinking" ? Math.sin(time * 1.1) * 0.012 : 0;
    a.mouthShift = smoothToward(a.mouthShift, expression.mouthShift + hmm, dt, 4);
    parts.mouthGroup.position.x = a.mouthShift;
    const flatness = clamp(1 - Math.abs(a.mouthCurve) / 0.35, 0, 1);
    const closed = clamp(1 - a.mouthOpen * 2.2, 0, 1);
    const curve = Math.abs(a.mouthCurve) < 0.05 ? 0.05 : a.mouthCurve;
    parts.mouthArc.scale.set(a.mouthWidth, curve, 1);
    parts.mouthArc.position.y = a.mouthCurve < 0 ? -0.035 : 0;
    parts.mouthArcMaterial.opacity = closed * (1 - flatness);
    parts.mouthLine.visible = flatness > 0.01;
    parts.mouthLine.scale.set(1, a.mouthWidth, 1);
    parts.mouthLineMaterial.opacity = closed * flatness;
    parts.mouthOpen.visible = a.mouthOpen > 0.02;
    parts.mouthOpen.scale.set(1.15 - a.mouthOpen * 0.3, 0.18 + a.mouthOpen * 0.95, 1);

    // Head: float, sway, breathe; per-expression tilt; follows the eyes; nods with the voice.
    a.bounce = smoothToward(a.bounce, expression.bounce || 1, dt, 3);
    a.speed = smoothToward(a.speed, stateLook.motion * a.bounce, dt, 2);
    a.phase += dt * a.speed;
    const float = Math.sin(a.phase * 1.3) * 0.045;
    const sway = Math.sin(a.phase * 0.7) * 0.06;
    const breathe = 1 + Math.sin(a.phase * 2.1) * 0.008;
    a.roll = smoothToward(a.roll, expression.headRoll, dt, 3);
    a.headPitch = smoothToward(a.headPitch, expression.headPitch, dt, 3);
    const lean = state === "listening" ? 0.04 + level * 0.04 : 0;
    a.followX = smoothToward(a.followX, a.gazeX * 2.2, dt, 4);
    a.followY = smoothToward(a.followY, -a.gazeY * 1.4, dt, 4);
    a.nod = smoothToward(a.nod, state === "speaking" ? level * 0.05 : 0, dt, 10);
    const shake = look === "error" && inLook < 0.7 ? Math.sin(inLook * 26) * 0.05 * (1 - inLook / 0.7) : 0;
    body.position.y = float + a.nod * 0.3;
    body.rotation.set(lean + a.headPitch + a.followY + a.nod, sway + a.followX + shake, sway * 0.25 + a.roll);
    body.scale.setScalar(breathe);
    a.halo = smoothToward(a.halo, 0.25 + stateLook.glow * 0.3 + level * 0.15, dt, 4);
    backHalo.uniforms.uStrength.value = a.halo;
    backHalo.uniforms.uTint.value.copy(a.tint);
    if (!activeMood() && signals.mood) { signals.mood = null; updateLabel(); }   // the mood has faded
  }

  function resize() {
    const width = container.clientWidth || 300, height = container.clientHeight || 240;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  // The render loop pauses while the tab is hidden.
  let running = true, rafId = 0;
  function loop() {
    if (!running) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    frame(dt, clock.getElapsedTime());
    renderer.render(scene, camera);
    rafId = requestAnimationFrame(loop);
  }
  const onVisibility = () => {
    if (document.hidden) { running = false; cancelAnimationFrame(rafId); }
    else if (!running) { running = true; clock.getDelta(); loop(); }
  };
  document.addEventListener("visibilitychange", onVisibility);
  loop();

  return {
    unmount() {
      running = false; cancelAnimationFrame(rafId);
      off.forEach((stop) => stop());
      observer.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      renderer.dispose();
      container.textContent = "";
    },
  };
}
