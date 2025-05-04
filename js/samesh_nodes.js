import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Helper function to add the upload widget
function addUploadWidget(nodeType, nodeData, widgetName) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        if (!pathWidget) {
            console.error(`SamMesh: Could not find widget '${widgetName}' on node ${nodeData.name}`);
            return;
        }

        const fileInput = document.createElement("input");
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });

        Object.assign(fileInput, {
            type: "file",
            // Define accepted file types (adjust if needed for samesh compatibility)
            accept: ".obj,.glb,.gltf,.ply,.stl,.3mf,model/obj,model/gltf-binary,model/gltf+json,application/vnd.ms-pki.stl,application/x-stl,application/vnd.ms-package.3dmanufacturing-3dmodel+xml,application/x-ply,application/ply",
            style: "display: none",
            onchange: async () => {
                if (fileInput.files.length) {
                    let resp = await uploadFile(fileInput.files[0]); // Use the uploadFile function below
                    if (!resp || resp.status !== 200) {
                        // Upload failed
                        return;
                    }
                    const filename = (await resp.json()).name;

                    // Check if the filename already exists in the dropdown
                    if (!pathWidget.options.values.includes(filename)) {
                        pathWidget.options.values.push(filename);
                    }
                    pathWidget.value = filename; // Set the widget value to the uploaded file
                    if (pathWidget.callback) {
                        pathWidget.callback(filename); // Trigger any callback associated with the widget
                    }
                     // Optional: Force node redraw if necessary
                    app.graph.setDirtyCanvas(true, true);
                }
            },
        });

        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose mesh file to upload", "mesh", () => { // Changed button text slightly
            // Clear the active click event
            app.canvas.node_widget = null;
            fileInput.click();
        });
        uploadWidget.options.serialize = false; // Don't save the button state
    });
}

// Helper function to upload the file to the ComfyUI server
async function uploadFile(file) {
    try {
        const body = new FormData();
        // Determine subfolder (if any) - using ComfyUI's logic if applicable
        // For simplicity, uploading directly to input for now.
        // const subfolder = file.webkitRelativePath ? file.webkitRelativePath.slice(0, file.webkitRelativePath.lastIndexOf('/') + 1) : "";
        const new_file = new File([file], file.name, { type: file.type, lastModified: file.lastModified });
        body.append("image", new_file); // ComfyUI's upload endpoint expects "image"
        body.append("overwrite", "true"); // Optional: overwrite existing file

        // Use ComfyUI's API endpoint for uploading
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            console.log("SamMesh: File uploaded successfully:", (await resp.clone().json()).name); // Clone response to read json multiple times if needed
            return resp;
        } else {
            alert(`SamMesh Upload Error: ${resp.status} - ${resp.statusText}`);
            return null;
        }
    } catch (error) {
        alert(`SamMesh Upload Exception: ${error}`);
        return null;
    }
}

// Helper function to chain callbacks
function chainCallback(object, property, callback) {
    if (object === undefined) {
        console.error("SamMesh: Tried to add callback to non-existant object");
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r;
        };
    } else {
        object[property] = callback;
    }
}


// Register the extension with ComfyUI
app.registerExtension({
    name: "SamMesh.jsnodes", // Unique name for your extension
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Check if the node is in the "SamMesh" category
        if (!nodeData?.category?.startsWith("SamMesh")) {
            return;
        }

        // Add widgets or modify nodes based on their name
        switch (nodeData.name) {
            case "SamMeshLoader":
                // Add the upload widget to the 'mesh' dropdown
                addUploadWidget(nodeType, nodeData, "mesh");
                break;
            // Add cases for other SamMesh nodes if needed in the future
        }
    },
}); 