import { FastifyRequest, FastifyReply } from "fastify";
import { getPrediction } from "../../getPrediction.js";
import { FromSchema } from "json-schema-to-ts";
import axios from "axios";

export const singleMultipartFormBodySchema = {
	type: "object",
	properties: {
		content: {
			type: "array",
			items: {
				$ref: "#mySharedSchema",
			},
		},
	},
	required: ["content"],
} as const;

type BodyEntry = {
	data: Buffer;
	filename: string;
	encoding: string;
	mimetype: string;
	limit: false;
};

export async function SingleMultipartForm(
	request: FastifyRequest<{
		Body: FromSchema<typeof singleMultipartFormBodySchema>;
	}>,
	reply: FastifyReply,
) {
	const image = request.body.content[0] as BodyEntry;

	let imageData;
	if (isValidUrl(image.filename)) {
	  const response = await axios.get(image.filename, { responseType: "arraybuffer" });
	  imageData = Buffer.from(response.data);
	} else {
	  imageData = image.data;
	}

	return reply.send({
		prediction: await getPrediction(imageData),
	});

	function isValidUrl(string) {
	  try {
		new URL(string);
		return true;
	  } catch (_) {
		return false;
	  }
	}
}
