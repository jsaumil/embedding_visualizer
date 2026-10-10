import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Force the browser entry of @huggingface/transformers. The package
      // has a "node" condition that wins in a Next.js build, but the Node
      // entry pulls in WASM/onnxruntime paths that don't exist in the
      // client bundle. Strip the "node" condition from the client build's
      // resolver so the "default" (browser) entry is used instead.
      config.resolve.alias = {
        ...(config.resolve.alias || {}),
        "@huggingface/transformers": path.resolve(
          __dirname,
          "node_modules/@huggingface/transformers/dist/transformers.web.js",
        ),
      };
      // Also ensure webpack doesn't try to resolve WASM/worker files.
      config.module.rules.push({
        test: /\.wasm$/,
        loader: "ignore-loader",
      });
      config.resolve.fullySpecified = false;
    }
    return config;
  },
};

export default nextConfig;