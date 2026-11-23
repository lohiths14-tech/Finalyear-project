from locust import HttpUser, task, between
import io

class AgriVisionBenchmarkUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def health_check(self):
        self.client.get("/api/v1/health")
        
    @task
    def ready_check(self):
        self.client.get("/api/v1/ready")

    @task(3)
    def analyze_leaf(self):
        # Create a dummy image
        file_obj = io.BytesIO(b"dummy image content, enough to pass initial size check but will fail decoding if not a real image. We'll use a tiny valid gif.")
        # Actually, let's use a 1x1 pixel PNG
        tiny_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
            b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        file_obj = io.BytesIO(tiny_png)
        
        self.client.post(
            "/api/v1/analyze/leaf",
            files={"file": ("test.png", file_obj, "image/png")}
        )

    @task(1)
    def analyze_field(self):
        tiny_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
            b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        file_obj = io.BytesIO(tiny_png)
        
        self.client.post(
            "/api/v1/analyze/field",
            files={"file": ("field_test.png", file_obj, "image/png")},
            data={"enable_geospatial": "false"}
        )
