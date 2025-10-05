#!/usr/bin/env python3
"""
Test script to verify the weather prediction model works correctly
"""

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        from weather_prediction_model import WeatherPredictionModel
        print("✅ WeatherPredictionModel imported successfully")
        
        # Test dependency check
        status = WeatherPredictionModel.check_dependencies()
        print(f"✅ Dependencies status: {status}")
        
        # Try to create model instance
        model = WeatherPredictionModel()
        print("✅ Model instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_weather_data_collector():
    """Test weather data collector"""
    print("\nTesting weather data collector...")
    
    try:
        from weather_data_collector import WeatherDataCollector
        collector = WeatherDataCollector()
        print("✅ WeatherDataCollector imported and created successfully")
        
        # Test coordinate lookup
        lat, lon = collector.get_coordinates("Toronto")
        print(f"✅ Toronto coordinates: {lat}, {lon}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weather data collector test failed: {e}")
        return False

def test_notebook_integration():
    """Test notebook integration"""
    print("\nTesting notebook integration...")
    
    try:
        from notebook_integration import NotebookExecutor
        executor = NotebookExecutor()
        print("✅ NotebookExecutor imported and created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook integration test failed: {e}")
        return False

def test_core_backend():
    """Test core backend functionality"""
    print("\nTesting core backend...")
    
    try:
        import petrichor_backend
        print("✅ Main backend module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Core backend test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PETRICHOR BACKEND COMPONENT TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_weather_data_collector,
        test_notebook_integration,
        test_core_backend
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your backend is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    print("=" * 60)