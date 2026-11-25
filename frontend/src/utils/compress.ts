/**
 * Client-side image optimizer for fast cloud inference and low bandwidth.
 * Resizes images larger than 1024px to max 1024x1024 and compresses to JPEG.
 */
export async function optimizeImageForUpload(file: File, maxDim = 1024): Promise<File> {
  // If not an image or already very small (<100KB), return original immediately
  if (!file.type.startsWith('image/') || file.size < 100 * 1024) {
    return file;
  }

  return new Promise((resolve) => {
    const safetyTimer = setTimeout(() => resolve(file), 3000);
    const done = (out: File) => {
      clearTimeout(safetyTimer);
      resolve(out);
    };

    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (e) => {
      const img = new Image();
      img.src = e.target?.result as string;
      img.onload = () => {
        let { width, height } = img;

        // If already within bounds, don't re-compress if size is under 500KB
        if (width <= maxDim && height <= maxDim && file.size < 500 * 1024) {
          done(file);
          return;
        }

        // Calculate aspect ratio downscaling
        if (width > maxDim || height > maxDim) {
          if (width > height) {
            height = Math.round((height * maxDim) / width);
            width = maxDim;
          } else {
            width = Math.round((width * maxDim) / height);
            height = maxDim;
          }
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          done(file);
          return;
        }

        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const cleanName = file.name.replace(/\.[^.]+$/, '') + '.jpg';
              done(new File([blob], cleanName, { type: 'image/jpeg' }));
            } else {
              done(file);
            }
          },
          'image/jpeg',
          0.85
        );
      };
      img.onerror = () => done(file);
    };
    reader.onerror = () => done(file);
  });
}
