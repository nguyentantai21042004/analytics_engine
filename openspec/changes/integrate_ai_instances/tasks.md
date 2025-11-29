# Implementation Tasks

**Change ID**: `integrate_ai_instances`  
**Estimated Effort**: 6-8 hours

---

## Phase 1: Service Lifecycle Integration (3 hours) ✅ COMPLETED

### 1.1 Update API Service Lifespan ✅
- [x] Modify `commands/api/main.py` lifespan context manager
- [x] Add PhoBERT initialization in startup
- [x] Add SpaCy-YAKE initialization in startup
- [x] Add proper error handling for model loading failures
- [x] Add logging for model lifecycle events
- [x] Store models in `app.state`
- [x] Add cleanup in shutdown

**Acceptance Criteria**: ✅ ALL MET
- ✅ Models initialize successfully on startup
- ✅ Startup completes in < 10 seconds
- ✅ Proper error messages if models fail to load
- ✅ Models properly cleaned up on shutdown

### 1.2 Implement Infrastructure Layer ✅
- [x] Create `infrastructure/messaging/` directory
- [x] Create `infrastructure/messaging/rabbitmq.py`
- [x] Implement `RabbitMQClient` class
- [x] Add `connect()` method with robust connection
- [x] Add `close()` method
- [x] Add `consume()` method with QoS setup
- [ ] Add unit tests for `RabbitMQClient` (deferred to Phase 4)

### 1.3 Update Consumer Service Lifecycle ✅
- [x] Add `aio-pika` dependency (made optional with graceful handling)
- [x] Add RabbitMQ configuration to `core/config.py`
- [x] Modify `commands/consumer/main.py`
- [x] Initialize AI models
- [x] Initialize `RabbitMQClient`
- [x] Create message handler wrapper
- [x] Start consumption loop
- [x] Implement graceful shutdown

**Acceptance Criteria**: ✅ ALL MET
- ✅ Infrastructure layer handles all RabbitMQ logic
- ✅ Consumer service is clean (only orchestration)
- ✅ Models passed correctly to handlers
- ✅ Graceful shutdown works correctly

---

## Phase 2: Dependency Injection (1 hour)

### 2.1 Create Dependencies Module
- [ ] Create `internal/api/dependencies.py`
- [ ] Implement `get_phobert()` dependency function
- [ ] Implement `get_spacyyake()` dependency function
- [ ] Add type hints and docstrings
- [ ] Add error handling for missing models

**Acceptance Criteria**:
- Dependencies return correct model instances
- Type hints work correctly with IDE autocomplete
- Proper error if models not initialized

---

## Phase 3: Test API Endpoint (2 hours)

### 3.1 Create Test Router
- [ ] Create `internal/api/routes/test.py`
- [ ] Define `AnalyticsTestRequest` Pydantic model
- [ ] Define `AnalyticsTestResponse` Pydantic model
- [ ] Implement `/api/v1/test/analytics` POST endpoint
- [ ] Add dependency injection for models
- [ ] Add input validation
- [ ] Add error handling
- [ ] Add logging

### 3.2 Register Router
- [ ] Update `internal/api/main.py` to include test router
- [ ] Verify OpenAPI docs generation
- [ ] Test endpoint via Swagger UI

**Acceptance Criteria**:
- Endpoint accepts JSON matching master-proposal.md format
- Endpoint returns full analytics debug response
- Response time < 1 second
- Proper error messages for invalid input
- Endpoint visible in Swagger UI

---

## Phase 4: Testing (1 hour)

### 4.1 Unit Tests
- [ ] Test `get_phobert()` dependency
- [ ] Test `get_spacyyake()` dependency
- [ ] Test endpoint with valid JSON
- [ ] Test endpoint with invalid JSON
- [ ] Test endpoint when models not initialized

### 4.2 Integration Tests
- [ ] Test full API startup with models
- [ ] Test endpoint with real JSON from master-proposal.md
- [ ] Verify models are reused across requests (not reloaded)
- [ ] Test shutdown cleanup

**Acceptance Criteria**:
- All tests passing
- Test coverage > 80%
- Integration test validates full flow

---

## Phase 5: Documentation (30 minutes)

### 5.1 Update Documentation
- [ ] Add usage examples to README.md
- [ ] Document test endpoint in API docs
- [ ] Add curl examples for testing
- [ ] Update project.md with integration details

**Acceptance Criteria**:
- Clear examples of how to use test endpoint
- Documentation matches implementation

---

## Success Metrics

- [ ] API starts successfully with models loaded
- [ ] Startup time < 10 seconds
- [ ] Test endpoint response time < 1 second
- [ ] All tests passing (unit + integration)
- [ ] No memory leaks (models properly cleaned up)
- [ ] OpenAPI docs generated correctly
