class GlassesRecommendationEngine:
    """Provide glasses recommendations based on face shape"""
    
    def __init__(self):
        self.recommendations = {
            'heart': {
                'suitable': ['Round frames', 'Oval frames', 'Cat-eye', 'Rimless'],
                'avoid': ['Heavy top frames', 'Geometric shapes'],
                'tip': 'Choose frames that balance wider forehead',
                'description': 'Heart-shaped faces have wider foreheads and narrow chins'
            },
            'oblong': {
                'suitable': ['Oversized frames', 'Square frames', 'Wide frames', 'Decorative temples'],
                'avoid': ['Small frames', 'Narrow styles'],
                'tip': 'Wider frames help balance face length',
                'description': 'Oblong faces are longer than they are wide'
            },
            'oval': {
                'suitable': ['Most styles work!', 'Square', 'Rectangular', 'Geometric'],
                'avoid': ['Oversized frames that hide features'],
                'tip': 'Lucky you! Most styles complement oval faces',
                'description': 'Oval faces have balanced proportions'
            },
            'round': {
                'suitable': ['Angular frames', 'Rectangular', 'Square', 'Cat-eye'],
                'avoid': ['Round frames', 'Small frames'],
                'tip': 'Angular frames add definition to soft curves',
                'description': 'Round faces have full cheeks and soft curves'
            },
            'square': {
                'suitable': ['Round frames', 'Oval frames', 'Curved styles', 'Aviators'],
                'avoid': ['Angular frames', 'Boxy squares'],
                'tip': 'Softer curves balance strong jawline',
                'description': 'Square faces have strong jawlines and broad foreheads'
            }
        }
    
    def get_recommendation(self, face_shape):
        """Get glasses recommendation for given face shape"""
        return self.recommendations.get(face_shape.lower(), {
            'suitable': ['Consult an optician'],
            'avoid': [],
            'tip': 'Face shape not recognized',
            'description': ''
        })
    
    def print_recommendations(self, face_shape):
        """Print detailed recommendations to console"""
        rec = self.get_recommendation(face_shape)
        print(f"\n{'='*50}")
        print(f"Face Shape: {face_shape.upper()}")
        print(f"{'='*50}")
        print(f"\nDescription: {rec['description']}")
        print(f"\n✓ Suitable frames:")
        for style in rec['suitable']:
            print(f"  • {style}")
        print(f"\n✗ Avoid:")
        for style in rec['avoid']:
            print(f"  • {style}")
        print(f"\n💡 Tip: {rec['tip']}")
        print(f"{'='*50}\n")
