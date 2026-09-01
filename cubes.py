"""
"""

# test_example = true

import wgpu

from rendercanvas.auto import RenderCanvas, loop
import numpy as np
from pyglm import glm
import time

# the shader code is provided as a string literal for portability
wgsl_shader_source = """
struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texcoord: vec2<f32>,
};

struct Locals {
    transform: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> r_locals: Locals;

struct VertexOutput {
    @location(0) color : vec4f,
    @builtin(position) pos: vec4f,
};

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) index: u32) -> VertexOutput {

    var out: VertexOutput;
    let x: f32 = f32(index % 100) - 50;
    let y: f32 = f32(index / 100) - 90;
    out.pos = r_locals.transform * (in.pos + vec4(3.0 * x, 0.0, - 3.0 * y, 1.0));
    out.color = vec4(in.texcoord, 1.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let physical_color = pow(in.color.rgb, vec3(2.2));  // gamma correct
    return vec4(physical_color, in.color.a);
}
"""

# pos         texcoord
# x, y, z, w, u, v
vertex_data = np.array(
    [
        # top (0, 0, 1)
        [-1, -1, 1, 1, 0, 0],
        [1, -1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [-1, 1, 1, 1, 0, 1],
        # bottom (0, 0, -1)
        [-1, 1, -1, 1, 1, 0],
        [1, 1, -1, 1, 0, 0],
        [1, -1, -1, 1, 0, 1],
        [-1, -1, -1, 1, 1, 1],
        # right (1, 0, 0)
        [1, -1, -1, 1, 0, 0],
        [1, 1, -1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, -1, 1, 1, 0, 1],
        # left (-1, 0, 0)
        [-1, -1, 1, 1, 1, 0],
        [-1, 1, 1, 1, 0, 0],
        [-1, 1, -1, 1, 0, 1],
        [-1, -1, -1, 1, 1, 1],
        # front (0, 1, 0)
        [1, 1, -1, 1, 1, 0],
        [-1, 1, -1, 1, 0, 0],
        [-1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
        # back (0, -1, 0)
        [1, -1, 1, 1, 0, 0],
        [-1, -1, 1, 1, 1, 0],
        [-1, -1, -1, 1, 1, 1],
        [1, -1, -1, 1, 0, 1],
    ],
    dtype=np.float32,
)

index_data = np.array(
    [
        [0, 1, 2, 2, 3, 0],  # top
        [4, 5, 6, 6, 7, 4],  # bottom
        [8, 9, 10, 10, 11, 8],  # right
        [12, 13, 14, 14, 15, 12],  # left
        [16, 17, 18, 18, 19, 16],  # front
        [20, 21, 22, 22, 23, 20],  # back
    ],
    dtype=np.uint32,
).flatten()


# adapter is required to create the device object, which is the general entry point to create most wgpu objects.
# for convenience and interoperability `wgpu.utils.get_default_device()` and associated configuration are provided.
adapter = wgpu.gpu.request_adapter_sync()
device = adapter.request_device_sync()

# setting up a canvas, so we can see what we draw
window_size = (640, 480)
canvas = RenderCanvas(size=window_size, title="wgpu triangle example", update_mode='continuous')
context = canvas.get_wgpu_context()
render_texture_format = context.get_preferred_format(device.adapter)
context.configure(device=device, format=render_texture_format)

# creating the shader module compiles the shader code for your GPU.
shader = device.create_shader_module(code=wgsl_shader_source)

uniform_buffer = device.create_buffer(
    size= 4 * 4 * 4, # 4x4 f32 matrix
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    label="Cube Example uniform buffer",
)

bind_group_layout = device.create_bind_group_layout(
    entries=[
        wgpu.BindGroupLayoutEntry(
            binding=0,
            visibility=wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
            buffer={},
        )
    ],
    label="Cube Example bind group layout",
)

bind_group = device.create_bind_group(
    layout=bind_group_layout,
    entries=[
        wgpu.BindGroupEntry(
            binding=0,
            resource=uniform_buffer,
        )
    ],
    label="Cube Example bind group",
)

pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout], label="Cube Example pipeline layout"
)

# in wgpu-py, methods that take descriptors will take the keyword arguments instead.
# descriptors and other structs can still be accessed via wgpu.structs or top level wgpu.
render_pipeline = device.create_render_pipeline(
    **wgpu.structs.RenderPipelineDescriptor(
        layout=pipeline_layout,
        vertex=wgpu.structs.VertexState(
            module=shader,
            buffers=[
                wgpu.VertexBufferLayout(
                    array_stride=4 * 6,
                    step_mode="vertex",
                    attributes=[
                        wgpu.VertexAttribute(
                            format="float32x4",
                            offset=0,
                            shader_location=0,
                        ),
                        wgpu.VertexAttribute(
                            format="float32x2",
                            offset=4 * 4,
                            shader_location=1,
                        ),
                    ],
                ),
            ],
        ),
        primitive=wgpu.PrimitiveState(
            topology="triangle-list",
            front_face="ccw",
            cull_mode="back",
        ),
        depth_stencil=None,
        multisample=None,
        fragment=wgpu.FragmentState(
            module=shader,
            targets=[wgpu.ColorTargetState(format=render_texture_format)],
        ),
    )
)


vertex_buffer = device.create_buffer_with_data(
    data=vertex_data,
    usage=wgpu.BufferUsage.VERTEX,
    label="Cube Example vertex buffer",
)

# Create index buffer, and upload data
index_buffer = device.create_buffer_with_data(
    data=index_data, usage=wgpu.BufferUsage.INDEX, label="Cube Example index buffer"
)

# this function gets called for every frame. It ends with submitting a buffer of work onto the GPU queue.
def drawing_function():
    command_encoder = device.create_command_encoder()
    current_texture_view = context.get_current_texture().create_view()

    t = time.time()

    model = glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(t % 360) * 20.0, glm.vec3(1.0, 0.3, 0.5))

    uniform_data = glm.perspectiveFovRH_ZO(glm.radians(60), window_size[0], window_size[1], 0.1, 100.0) * glm.lookAt(
            glm.vec3(3.0, 3.0, -6.0),
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 1.0, 0.0)) * model

    device.queue.write_buffer(uniform_buffer, 0, uniform_data.to_bytes())


    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=current_texture_view,
                clear_value=(0.5, 0.5, 1.0, 1),  # a green background
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
            )
        ],
    )
    render_pass.set_pipeline(render_pipeline)
    render_pass.set_index_buffer(index_buffer, "uint32")
    render_pass.set_vertex_buffer(0, vertex_buffer)
    render_pass.set_bind_group(0, bind_group)
    render_pass.draw_indexed(index_data.size, 10000)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])


canvas.request_draw(drawing_function)


if __name__ == "__main__":
    loop.run()