# Stage 1: Build the JAR
FROM maven:3.9-eclipse-temurin-21 AS builder
WORKDIR /app
COPY pom.xml .
COPY src ./src
# Build, skipping tests to save time
RUN mvn clean package -DskipTests

# Stage 2: Run the App (Tiny Image)
FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar

RUN mkdir -p /app/models
COPY src/main/resources/models/multiclass_classifier2.bin /app/models/multiclass_classifier2.bin
ENTRYPOINT ["java", "-jar", "app.jar"]