// RedsiaUtils.java
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class RedsiaUtils {

    /**
     * Pré-processa a imagem para a rede ONNX
     * Redimensiona para 224x224, central crop e normaliza
     */
    public static float[] preprocessImage(File imgFile) throws IOException {
        int size = 224;
        BufferedImage img = ImageIO.read(imgFile);

        // central crop
        int width = img.getWidth();
        int height = img.getHeight();
        int minDim = Math.min(width, height);
        int x = (width - minDim) / 2;
        int y = (height - minDim) / 2;
        BufferedImage cropped = img.getSubimage(x, y, minDim, minDim);

        // resize para 224x224
        BufferedImage resized = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(cropped, 0, 0, size, size, null);

        // normalize
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        float[] data = new float[3 * size * size];

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                int rgb = resized.getRGB(i, j);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                data[0 * size * size + j * size + i] = (r / 255.0f - mean[0]) / std[0];
                data[1 * size * size + j * size + i] = (g / 255.0f - mean[1]) / std[1];
                data[2 * size * size + j * size + i] = (b / 255.0f - mean[2]) / std[2];
            }
        }
        return data;
    }
}

