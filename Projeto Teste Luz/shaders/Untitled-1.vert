#include "../../ConstantBuffers/PerFrame.hlsli"
#include "../../Utils/DepthUtils.hlsli"
#include "../../Constants.hlsli"

Texture3D baseShapeLookup : register(t0);
Texture3D erosionLookup : register(t1);
Texture2D weatherLookup : register(t2);
Texture2D depthTexture : register(t3);

SamplerState texSampler : register(s0);

cbuffer cbVolumetricClouds : register(b0)
{
	float3 cb_lightDirection; // light direction world space
	float cb_groundRadius; // meters - for a ground/lower atmosphere only version, this could be smaller
			 
	float3 cb_sunColor; // color of sun light
	uint cb_baseShapeTextureBottomMipLevel;

	float cb_cloudVolumeStartHeight; // meters - height above ground level
	float cb_cloudVolumeHeight; // meters - height above ground level
	float cb_cloudSpeed;
	float cb_cloudTopOffset;

	float3 cb_windDirection;
	uint cb_erosionTextureBottomMipLevel;

	float3 cb_weatherTexMod; // scale(x), offset(y, z)
	float cb_windStrength;
};

static const float VOLUME_END_HEIGHT = cb_cloudVolumeStartHeight + cb_cloudVolumeHeight;
// planet center (world space)
static const float3 PLANET_CENTER = float3(0.0f, -cb_groundRadius - 100.0f, 0.0f); // TODO revisit - 100.0f offset is to match planet sky settings
// radius from the planet center to the bottom of the cloud volume
static const float PLANET_CENTER_TO_LOWER_CLOUD_RADIUS = cb_groundRadius + cb_cloudVolumeStartHeight;
// radius from the planet center to the top of the cloud volume
static const float PLANET_CENTER_TO_UPPER_CLOUD_RADIUS = cb_groundRadius + VOLUME_END_HEIGHT;
static const float CLOUD_SCALE = 1.0f / VOLUME_END_HEIGHT;
static const float3 WEATHER_TEX_MOD = float3(1.0f / (VOLUME_END_HEIGHT * cb_weatherTexMod.x), cb_weatherTexMod.y, cb_weatherTexMod.z); 
static const float2 WEATHER_TEX_MOVE_SPEED = float2(cb_windStrength * cb_windDirection.x, cb_windStrength * cb_windDirection.z); // this is modded by app run time
// samples based on shell thickness between inner and outer volume
static const uint2 SAMPLE_RANGE = uint2(64u, 128u);

static const float4 STRATUS_GRADIENT = float4(0.02f, 0.05f, 0.09f, 0.11f);
static const float4 STRATOCUMULUS_GRADIENT = float4(0.02f, 0.2f, 0.48f, 0.625f);
static const float4 CUMULUS_GRADIENT = float4(0.01f, 0.0625f, 0.78f, 1.0f); // these fractions would need to be altered if cumulonimbus are added to the same pass

/**
 * Perform a ray-sphere intersection test.
 * Returns the number of intersections in the direction of the ray (excludes intersections behind the ray origin), between 0 and 2.
 * In the case of more than one intersection, the nearest point will be returned in t1.
 * 
 * http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
 */
uint intersectRaySphere(
	float3 rayOrigin,
	float3 rayDir, // must be normalized
	float3 sphereCenter,
	float  sphereRadius,
	out float2 t)
{
	float3 l = rayOrigin - sphereCenter;
	float a = 1.0f; // dot(rayDir, rayDir) where rayDir is normalized
	float b = 2.0f * dot(rayDir, l);
	float c = dot(l, l) - sphereRadius * sphereRadius;
	float discriminate = b * b - 4.0f * a * c;
	if(discriminate < 0.0f)
	{
		t.x = t.y = 0.0f;
		return 0u;
	}
	else if(abs(discriminate) - 0.00005f <= 0.0f)
	{
		t.x = t.y = -0.5f * b / a;
		return 1u;
	}
	else
	{
		float q = b > 0.0f ?
			-0.5f * (b + sqrt(discriminate)) :
			-0.5f * (b - sqrt(discriminate));
		float h1 = q / a;
		float h2 = c / q;
		t.x = min(h1, h2);
		t.y = max(h1, h2);
		if(t.x < 0.0f)
		{
			t.x = t.y;
			if(t.x < 0.0f)
			{
				return 0u;
			}
			return 1u;
		}
		return 2u;
	}
}

float remap(
	float value,
	float oldMin,
	float oldMax,
	float newMin,
	float newMax)
{
	return newMin + (value - oldMin) / (oldMax - oldMin) * (newMax - newMin);
}

float3 sampleWeather(float3 pos)
{
	return weatherLookup.SampleLevel(texSampler, pos.xz * WEATHER_TEX_MOD.x + WEATHER_TEX_MOD.yz + (WEATHER_TEX_MOVE_SPEED * cb_appRunTime), 0.0f).rgb;
}

float getCoverage(float3 weatherData)
{
	return weatherData.r;
}

float getPrecipitation(float3 weatherData)
{
	return weatherData.g;
}

float getCloudType(float3 weatherData)
{
	// weather b channel tells the cloud type 0.0 = stratus, 0.5 = stratocumulus, 1.0 = cumulus
	return weatherData.b;
}

float heightFraction(float3 pos)
{
	return saturate((distance(pos, PLANET_CENTER) - PLANET_CENTER_TO_LOWER_CLOUD_RADIUS) / cb_cloudVolumeHeight);
}

float4 mixGradients(
	float cloudType)
{
	float stratus = 1.0f - saturate(cloudType * 2.0f);
	float stratocumulus = 1.0f - abs(cloudType - 0.5f) * 2.0f;
	float cumulus = saturate(cloudType - 0.5f) * 2.0f;
	return STRATUS_GRADIENT * stratus + STRATOCUMULUS_GRADIENT ﻿* stratocumulus + CUMULUS_GRADIENT * cumulus;
}

float densityHeightGradient(
	float heightFrac,
	float cloudType)
{
	float4 cloudGradient = mixGradients(cloudType);
	return smoothstep(cloudGradient.x, cloudGradient.y, heightFrac) - smoothstep(cloudGradient.z, cloudGradient.w, heightFrac);
}

float sampleCloudDensity(
	float3 pos,
	float3 weatherData,
	float heightFrac,
	float lod)
{
	pos += heightFrac * cb_windDirection * cb_cloudTopOffset;
	pos += (cb_windDirection + float3(0.0f, -0.25f, 0.0f)) * cb_cloudSpeed * (cb_appRunTime/* * 25.0f*/); // the * 25.0f is just for testing to make the effect obvious
	pos *= CLOUD_SCALE;
	
	float4 lowFreqNoise = baseShapeLookup.SampleLevel(texSampler, pos, lerp(0.0f, cb_baseShapeTextureBottomMipLevel, lod));
	float lowFreqFBM =
		(lowFreqNoise.g * 0.625f) +
		(lowFreqNoise.b * 0.25f) +
		(lowFreqNoise.a * 0.125f);

	float baseCloud = remap(
		lowFreqNoise.r,
		-(1.0f - lowFreqFBM), 1.0f, // gets about the same results just using -lowFreqFBM
		0.0f, 1.0f);

	float densityGradient = densityHeightGradient(heightFrac, getCloudType(weatherData));
	baseCloud *= densityGradient;

	float cloudCoverage = getCoverage(weatherData);
	float baseCloudWithCoverage = remap(
		baseCloud,
		1.0f - cloudCoverage, 1.0f,
		0.0f, 1.0f);
	baseCloudWithCoverage *= cloudCoverage;

	//// TODO add curl noise
	//// pos += curlNoise.xy * (1.0f - heightFrac);

	float3 highFreqNoise = erosionLookup.SampleLevel(texSampler, pos * 0.1f, lerp(0.0f, cb_erosionTextureBottomMipLevel, lod)).rgb;
	float highFreqFBM =
		(highFreqNoise.r * 0.625f) +
		(highFreqNoise.g * 0.25f) +
		(highFreqNoise.b * 0.125f);

	float highFreqNoiseModifier = lerp(highFreqFBM, 1.0f - highFreqFBM, saturate(heightFrac * 10.0f));

	baseCloudWithCoverage = remap(
		baseCloudWithCoverage,
		highFreqNoiseModifier * 0.2f, 1.0f,
		0.0f, 1.0f);

	return saturate(baseCloudWithCoverage);
}

struct VertexOut
{
	float4 posH : SV_POSITION;
	float3 viewRay : VIEWRAY;
	float2 tex : TEXCOORD;
};

// random vectors on the unit sphere
static const float3 RANDOM_VECTORS[] =
{
	float3( 0.38051305f,  0.92453449f, -0.02111345f),
	float3(-0.50625799f, -0.03590792f, -0.86163418f),
	float3(-0.32509218f, -0.94557439f,  0.01428793f),
	float3( 0.09026238f, -0.27376545f,  0.95755165f),
	float3( 0.28128598f,  0.42443639f, -0.86065785f),
	float3(-0.16852403f,  0.14748697f,  0.97460106f)
};

static const uint LIGHT_RAY_ITERATIONS = 6u;
static const float RCP_LIGHT_RAY_ITERATIONS = 1.0f / float(LIGHT_RAY_ITERATIONS);

float beerLambert(float sampleDensity, float precipitation)
{
	return exp(-sampleDensity * precipitation);
}

float powder(float sampleDensity, float lightDotEye)
{
	float powd = 1.0f - exp(-sampleDensity * 2.0f);
	return lerp(
		1.0f,
		powd,
		saturate((-lightDotEye * 0.5f) + 0.5f) // [-1,1]->[0,1]
	);
}

float henyeyGreenstein(
	float lightDotEye,
	float g)
{
	float g2 = g * g;
	return ((1.0f - g2) / pow((1.0f + g2 - 2.0f * g * lightDotEye), 1.5f)) * 0.25f;
}

float lightEnergy(
	float lightDotEye,
	float densitySample,
	float originalDensity,
	float precipitation)
{
	return 2.0f *
		beerLambert(densitySample, precipitation) *
		powder(originalDensity, lightDotEye) * 
		lerp(henyeyGreenstein(lightDotEye, 0.8f), henyeyGreenstein(lightDotEye, -0.5f), 0.5f);
}

// TODO get from cb values - has to change as time of day changes
float3 ambientLight(float heightFrac)
{
	return lerp(
		float3(0.5f, 0.67f, 0.82f),
		float3(1.0f, 1.0f, 1.0f),
		heightFrac);
}

float sampleCloudDensityAlongCone(
	float3 startPos,
	float  stepSize,
	float  lightDotEye,
	float  originalDensity)
{
	float3 lightStep = stepSize * -cb_lightDirection;
	float3 pos = startPos;
	float coneRadius = 1.0f;
	float coneStep = RCP_LIGHT_RAY_ITERATIONS;
	float densityAlongCone = 0.0f;﻿
	float lod = 0.0f;
	float lodStride = RCP_LIGHT_RAY_ITERATIONS;
	float3 weatherData = 0.0f;
	float rcpThickness = 1.0f / (stepSize * LIGHT_RAY_ITERATIONS);
	float density = 0.0f;

	for(uint i = 0u; i < LIGHT_RAY_ITERATIONS; ++i)
	{
		float3 conePos = pos + coneRadius * RANDOM_VECTORS[i] * float(i + 1u);
		float heightFrac = heightFraction(conePos);
		if(heightFrac <= 1.0f)
		{
			weatherData = sampleWeather(conePos);
			float cloudDensity = sampleCloudDensity(
				conePos,
				weatherData,
				heightFrac,
				lod);
			if(cloudDensity > 0.0f)
			{
				density += cloudDensity;
				float transmittance = 1.0f - (density * rcpThickness);
				densityAlongCone += (cloudDensity * transmittance);
			}
		}
		pos += lightStep;
		coneRadius += coneStep;
		lod += lodStride;
	}
	// take additional step at large distance away for shadowing from other clouds
	pos = pos + (lightStep * 8.0f);
	weatherData = sampleWeather(pos);
	float heightFrac = heightFraction(pos);
	if(heightFrac <= 1.0f)
	{
		float cloudDensity = sampleCloudDensity(
			pos,
			weatherData,
			heightFrac,
			0.8f);
		// no need to branch here since density variable is no longer used after this
		density += cloudDensity;
		float transmittance = 1.0f - saturate(density * rcpThickness);
		densityAlongCone += (cloudDensity * transmittance);
	}

	return saturate(lightEnergy(
		lightDotEye,
		densityAlongCone,
		originalDensity,
		lerp(1.0f, 2.0f, getPrecipitation(weatherData))));
}

float4 traceClouds(
	float3 viewDirW,		// world space view direction
	float3 startPos,		// world space start position
	float3 endPos)			// world space end position
{
	float3 dir = endPos - startPos;
	float thickness = length(dir);
	float rcpThickness = 1.0f / thickness;
	uint sampleCount = lerp(SAMPLE_RANGE.x, SAMPLE_RANGE.y, saturate((thickness - cb_cloudVolumeHeight) / cb_cloudVolumeHeight));
	float stepSize = thickness / float(sampleCount);
	dir /= thickness;
	float3 posStep = stepSize * dir;

	float lightDotEye = -dot(cb_lightDirection, viewDirW);

	float3 pos = startPos;
	float3 weatherData = 0.0f;
	float4 result = 0.0f;
	float density = 0.0f;
	
	for(uint i = 0u; i < sampleCount; ++i)
	{
		float heightFrac = heightFraction(pos);
		weatherData = sampleWeather(pos);
		float cloudDensity = sampleCloudDensity(
			pos,
			weatherData,
			heightFrac,
			0.0f);

		if(cloudDensity > 0.0f)
		{
			density += cloudDensity;
			float transmittance = 1.0f - (density * rcpThickness);
			float lightDensity = sampleCloudDensityAlongCone(
				pos,
				stepSize,
				lightDotEye,
				cloudDensity);

			float3 ambientBadApprox = ambientLight(heightFrac) * min(1.0f, length(cb_sunColor.rgb * 0.0125f)) * transmittance;
			float4 source = float4((cb_sunColor.rgb * lightDensity) + ambientBadApprox/*+ ambientLight(heightFrac)*/, cloudDensity * transmittance); // TODO enable ambient when added to constant buffer
			source.rgb *= source.a;
			result = (1.0f - result.a) * source + result;
			if(result.a >= 1.0f) break;
		}

		pos += posStep;
	}

	// experimental fog - may not be needed if clouds are drawn before atmosphere - would have to draw sun by itself, then clouds, then atmosphere
	// fogAmt = 0 to disable
	float fogAmt = 1.0f - exp(-distance(startPos, cb_eyePositionW) * 0.00001f);
	float3 fogColor = float3(0.3f, 0.4f, 0.45f) * length(cb_sunColor.rgb * 0.125f) * 0.8f;
	float3 sunColor = normalize(cb_sunColor.rgb) * 4.0f * length(cb_sunColor.rgb * 0.125f);
	fogColor = lerp(fogColor, sunColor, pow(saturate(lightDotEye), 8.0f));
	return float4(clamp(lerp(result.rgb, fogColor, fogAmt), 0.0f, 1000.0f), saturate(result.a));
}

float4 main(VertexOut pIn) : SV_TARGET
{
	int3 loadIndices = int3(pIn.posH.xy, 0);
	float zwDepth = depthTexture.Load(loadIndices).r;
	bool depthPresent = zwDepth < 1.0f;
	float depth = linearizeDepth(zwDepth);
	float3 posV = pIn.viewRay * depth;
	float3 posW = mul(float4(posV, 1.0f), cb_inverseViewMatrix).xyz;
	float3 viewDirW = normalize(posW - cb_eyePositionW);

	// find nearest planet surface point
	float2 ph = 0.0f;
	uint planetHits = intersectRaySphere(
		cb_eyePositionW,
		viewDirW,
		PLANET_CENTER,
		cb_groundRadius,
		ph);

	// find nearest inner shell point
	float2 ih = 0.0f;
	uint innerShellHits = intersectRaySphere(
		cb_eyePositionW,
		viewDirW,
		PLANET_CENTER,
		PLANET_CENTER_TO_LOWER_CLOUD_RADIUS,
		ih);

	// find nearest outer shell point
	float2 oh = 0.0f;
	uint outerShellHits = intersectRaySphere(
		cb_eyePositionW,
		viewDirW,
		PLANET_CENTER,
		PLANET_CENTER_TO_UPPER_CLOUD_RADIUS,
		oh);

	// world space ray intersections
	float3 planetHitSpot = cb_eyePositionW + (viewDirW * ph.x);
	float3 innerShellHit = cb_eyePositionW + (viewDirW * ih.x);
	float3 outerShellHit = cb_eyePositionW + (viewDirW * oh.x);

	// eye radius from planet center
	float eyeRadius = distance(cb_eyePositionW, PLANET_CENTER);
	if(eyeRadius < PLANET_CENTER_TO_LOWER_CLOUD_RADIUS) // under inner shell
	{
		// exit if there's something in front of the start of the cloud volume
		if((depthPresent && (distance(posW, cb_eyePositionW) < distance(innerShellHit, cb_eyePositionW))) ||
			planetHits > 0u) // shell hits are guaranteed, but the ground may be occluding cloud layer
		{
			return float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
		return traceClouds(
			viewDirW,
			innerShellHit,
			outerShellHit);
	}
	else if(eyeRadius > PLANET_CENTER_TO_UPPER_CLOUD_RADIUS) // over outer shell
	{
		// possibilities are
		// 1) enter outer shell, leave inner shell
		// 2) enter outer shell, leave outer shell
		float3 firstShellHit = outerShellHit;
		if(outerShellHits == 0u ||
			depthPresent && (distance(posW, cb_eyePositionW) < distance(firstShellHit, cb_eyePositionW)))
		{
			return float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
		float3 secondShellHit = outerShellHits == 2u && innerShellHits == 0u ? cb_eyePositionW + (viewDirW * oh.y) : innerShellHit;
		return traceClouds(
			viewDirW,
			firstShellHit,
			secondShellHit);
	}
	else // between shells
	{
		/*
		 * From a practical viewpoint (properly scaled planet, atmosphere, etc.)
		 * only one shell will be hit.
		 * Start position is always eye position.
		 */
		float3 shellHit = innerShellHits > 0u ? innerShellHit : outerShellHit;
		// if there's something in the depth buffer that's closer, that's the end point
		if(depthPresent && (distance(posW, cb_eyePositionW) < distance(shellHit, cb_eyePositionW)))
		{
			shellHit = posW;
		}
		return traceClouds(
			viewDirW,
			cb_eyePositionW,
			shellHit);
	}
}