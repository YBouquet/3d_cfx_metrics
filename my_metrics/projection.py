import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mitsuba as mi
mi.set_variant("scalar_rgb")
import pdb
class Renderer():
    def __init__(self, rendering :str):
        self.rendering = rendering
        self.xml_head = \
        """
        <scene version="0.6.0">
            <integrator type="path">
                <integer name="maxDepth" value="-1"/>
            </integrator>
            <sensor type="perspective">
                <float name="farClip" value="100"/>
                <float name="nearClip" value="0.1"/>
                <transform name="toWorld">
                    <lookat origin="3,0,0" target="0,0,0" up="0,0,0.5"/>
                </transform>
                <float name="fov" value="25"/>
                
                <sampler type="ldsampler">
                    <integer name="sampleCount" value="256"/>
                </sampler>
                <film type="hdrfilm">
                    <integer name="width" value="1024"/>
                    <integer name="height" value="1024"/>
                    <rfilter type="gaussian"/>
                    <boolean name="banner" value="false"/>
                </film>
            </sensor>
            
            <bsdf type="roughplastic" id="surfaceMaterial">
                <string name="distribution" value="ggx"/>
                <float name="alpha" value="0.05"/>
                <float name="intIOR" value="1.46"/>
                <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
            </bsdf>
            
        """

        self.xml_ball_segment = \
        """
            <shape type="sphere">
                <float name="radius" value="0.025"/>
                <transform name="toWorld">
                    <translate x="{}" y="{}" z="{}"/>
                </transform>
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="{},{},{}"/>
                </bsdf>
            </shape>
        """

        self.xml_tail = \
        """
            <shape type="rectangle">
                <ref name="bsdf" id="surfaceMaterial"/>
                <transform name="toWorld">
                    <scale x="10" y="10" z="1"/>
                    <translate x="0" y="0" z="-0.5"/>
                </transform>
            </shape>
            
            <shape type="rectangle">
                <transform name="toWorld">
                    <scale x="10" y="10" z="1"/>
                    <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
                </transform>
                <emitter type="area">
                    <rgb name="radiance" value="1,1,1"/>
                </emitter>
            </shape>

        </scene>
        """

    def __standardize_bbox(self, pcd : np.ndarray):
        mins = np.amin(pcd, axis=0)
        maxs = np.amax(pcd, axis=0)
        center = ( mins + maxs ) / 2.
        scale = np.amax(maxs-mins)
        result = ((pcd - center)/scale).astype(np.float32) # [-0.5, 0.5]
        return result
    
    def __colormap(self, x,y,z):
        vec = np.array([x,y,z])
        vec = np.clip(vec, 0.001,1.0)
        norm = np.sqrt(np.sum(vec**2))
        vec /= norm
        return [vec[0], vec[1], vec[2]]

    def __render_mitsuba(self, pcd : np.array) -> np.array: #8UINT RGB
        xml_segments = [self.xml_head]
        pcd = self.__standardize_bbox(pcd)
        #pcd = pcd[:,[1,0,2]]

        for i in range(pcd.shape[0]):
            color = self.__colormap(pcd[i,0]+0.5,pcd[i,1]+0.5,pcd[i,2]+0.5-0.0125)
            xml_segments.append(self.xml_ball_segment.format(pcd[i,0],pcd[i,1],pcd[i,2], *color))
        xml_segments.append(self.xml_tail)

        xml_content = str.join('', xml_segments)
        # Load the scene
        scene = mi.load_string(xml_content)

        # Create an ImageBlock to store the rendered image
        image = mi.render(scene, spp=256)

        mybitmap = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
        image_np = np.array(mybitmap).clip(0,255) #.reshape((image.height(), image.width(), 3)).numpy()

        # Convert to PyTorch tensor
        return image_np
    
    def __render_matplotlib(self, pcd : np.array, elevation_angle: int = 90, azimuthal_angle: int = -90) -> np.array:
        # Load and standardize point cloud
        pcd = self.__standardize_bbox(pcd)

        # Create a 3D plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elevation_angle, azim=azimuthal_angle)

        # Plot each point as a sphere with color mapped to depth
        for i in range(pcd.shape[0]):
            color = self.__colormap(pcd[i, 0] + 0.5, pcd[i, 1] + 0.5, pcd[i, 2] + 0.5 - 0.0125)
            ax.scatter(pcd[i, 0], pcd[i, 1], pcd[i, 2], c=[color], marker='o', s=20)

        ax.axis("off")

        # Create a canvas to render the figure
        canvas = FigureCanvas(fig)
        canvas.draw()
        # Convert the figure to a tensor
        np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return np_image
    
    def apply(self, pcd : np.array):
        if self.rendering == "mitsuba":
            return self.__render_mitsuba(pcd)
        elif self.rendering == "matplotlib":
            return self.__render_matplotlib(pcd)
        else:
            raise NotImplementedError("Rendering method {} not implemented".format(self.rendering))
        